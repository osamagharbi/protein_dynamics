"""Networks of MAMBA-Flow.

Stage 1  ShapeVAE         time-free autoencoder of ONE surface frame.
         encoder: K anchor queries (fixed per protein, on its surface) cross-attend to N
                  surface points (Fourier position + normal), then self-attention -> per-anchor
                  Gaussian posterior, Z = K x d latent tokens.
         decoder: tokens (+ anchor position) -> self-attention -> a query point cross-attends
                  to the tokens -> signed distance. Trained with the IGR loss (DSR's loss).
Stage 2  SpaceTimeFlow    velocity field for flow matching over a T x K grid of latent tokens
         (T frames sorted by time, some observed = context, some generated = targets).
         block = adaLN( flow time, log frame spacing ) -> attention across the K tokens of a
                 frame -> bidirectional Mamba (forward + time-reversed scan) across the T
                 frames of each token -> MLP.

Attention runs through F.scaled_dot_product_attention when available (PyTorch >= 2.0),
except on the decoder's query path: the IGR loss differentiates the SDF w.r.t. the query
point and then backpropagates through that gradient, and the fused SDPA kernels have no
double-backward, so that path always uses the plain softmax(QK^T)V formulation.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

HAS_SDPA = hasattr(F, "scaled_dot_product_attention")


def attention(q, k, v, twice_differentiable=False):
    """q (B, H, Lq, D), k/v (B, H, Lk, D) -> (B, H, Lq, D)."""
    if HAS_SDPA and not twice_differentiable:
        return F.scaled_dot_product_attention(q, k, v)
    w = torch.softmax(torch.matmul(q * q.shape[-1] ** -0.5, k.transpose(-2, -1)), dim=-1)
    return torch.matmul(w.to(v.dtype), v)


class Fourier(nn.Module):
    """[x, sin(2^i pi x), cos(2^i pi x)], i = 0..n_freq-1 (coordinates in [-1.2, 1.2])."""

    def __init__(self, n_freq=8):
        super().__init__()
        self.register_buffer("freqs", (2.0 ** torch.arange(n_freq)) * math.pi, persistent=False)
        self.out_dim = 3 + 6 * n_freq

    def forward(self, x):
        xf = (x.unsqueeze(-1) * self.freqs).flatten(-2)
        return torch.cat([x, xf.sin(), xf.cos()], dim=-1)


def mlp(dim, ratio=4, dropout=0.0):
    return nn.Sequential(nn.Linear(dim, ratio * dim), nn.GELU(), nn.Dropout(dropout),
                         nn.Linear(ratio * dim, dim))


class MHA(nn.Module):
    def __init__(self, dim, heads, ctx_dim=None, twice_differentiable=False):
        super().__init__()
        assert dim % heads == 0, "width must be divisible by heads"
        self.h, self.twice = heads, twice_differentiable
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(ctx_dim or dim, 2 * dim)
        self.o = nn.Linear(dim, dim)

    def forward(self, x, ctx=None):
        ctx = x if ctx is None else ctx
        b, lq, c = x.shape
        lk = ctx.shape[1]
        q = self.q(x).view(b, lq, self.h, c // self.h).transpose(1, 2)
        k, v = self.kv(ctx).view(b, lk, 2, self.h, c // self.h).permute(2, 0, 3, 1, 4)
        out = attention(q, k, v, self.twice)
        return self.o(out.transpose(1, 2).reshape(b, lq, c))


class SelfBlock(nn.Module):
    def __init__(self, dim, heads, dropout=0.0):
        super().__init__()
        self.n1, self.attn = nn.LayerNorm(dim), MHA(dim, heads)
        self.n2, self.mlp = nn.LayerNorm(dim), mlp(dim, dropout=dropout)

    def forward(self, x):
        x = x + self.attn(self.n1(x))
        return x + self.mlp(self.n2(x))


class CrossBlock(nn.Module):
    def __init__(self, dim, heads, twice_differentiable=False):
        super().__init__()
        self.nq, self.nc = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = MHA(dim, heads, twice_differentiable=twice_differentiable)
        self.n2, self.mlp = nn.LayerNorm(dim), mlp(dim)

    def forward(self, x, ctx):
        x = x + self.attn(self.nq(x), self.nc(ctx))
        return x + self.mlp(self.n2(x))


# ----------------------------------------------------------------------------- Stage 1
class ShapeVAE(nn.Module):
    def __init__(self, width=384, latent_dim=32, heads=8, enc_layers=6, dec_layers=6,
                 n_freq=8, sphere_radius=0.6, query_blocks=1, head_layers=0, head_width=256):
        super().__init__()
        self.config = dict(width=width, latent_dim=latent_dim, heads=heads, enc_layers=enc_layers,
                           dec_layers=dec_layers, n_freq=n_freq, sphere_radius=sphere_radius,
                           query_blocks=query_blocks, head_layers=head_layers, head_width=head_width)
        self.sphere_radius = sphere_radius
        self.fourier = Fourier(n_freq)
        fd = self.fourier.out_dim
        self.point_in = nn.Linear(fd + 3, width)
        self.anchor_enc = nn.Linear(fd, width)
        self.enc_cross = CrossBlock(width, heads)
        self.enc_self = nn.ModuleList([SelfBlock(width, heads) for _ in range(enc_layers)])
        self.enc_out = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 2 * latent_dim))
        self.lat_in = nn.Linear(latent_dim, width)
        self.anchor_dec = nn.Linear(fd, width)
        self.dec_self = nn.ModuleList([SelfBlock(width, heads) for _ in range(dec_layers)])
        self.query_in = nn.Linear(fd, width)
        self.query_cross = nn.ModuleList([CrossBlock(width, heads, twice_differentiable=True)
                                          for _ in range(query_blocks)])
        self.out_norm = nn.LayerNorm(width)
        # optional per-query MLP head (depth per query point, cf. DSR's 8-layer MLP); it sees the
        # attention read-out and the query's Fourier features again
        layers, d_in = [], width + (fd if head_layers else 0)
        for _ in range(head_layers):
            layers += [nn.Linear(d_in, head_width), nn.GELU()]
            d_in = head_width
        self.head = nn.Sequential(*layers)
        self.out = nn.Linear(d_in, 1)
        nn.init.zeros_(self.out.weight)     # start from the sphere prior (cf. IGR geometric init)
        nn.init.zeros_(self.out.bias)

    def encode(self, pts, nrm, anchors):
        """pts/nrm (B, N, 3), anchors (B, K, 3) -> mu, logvar (B, K, d)."""
        x = self.point_in(torch.cat([self.fourier(pts), nrm], dim=-1))
        h = self.enc_cross(self.anchor_enc(self.fourier(anchors)), x)
        for blk in self.enc_self:
            h = blk(h)
        mu, raw = self.enc_out(h).chunk(2, dim=-1)
        # smooth clamp of log-variance to (-10, 4): unlike clamp() it never zeroes the gradient
        return mu, -3.0 + 7.0 * torch.tanh((raw + 3.0) / 7.0)

    def tokens(self, z, anchors):
        """Latents (B, K, d) -> decoder tokens (B, K, C); computed once per frame."""
        h = self.lat_in(z) + self.anchor_dec(self.fourier(anchors))
        for blk in self.dec_self:
            h = blk(h)
        return h

    def query(self, h, x):
        """Signed distance at query points x (B, Q, 3) given tokens h (B, K, C) -> (B, Q)."""
        fx = self.fourier(x)
        q = self.query_in(fx)
        for blk in self.query_cross:
            q = blk(q, h)
        q = self.out_norm(q)
        if len(self.head):
            q = self.head(torch.cat([q, fx.to(q.dtype)], dim=-1))
        sphere = torch.sqrt((x * x).sum(-1) + 1e-8) - self.sphere_radius
        return self.out(q).squeeze(-1).float() + sphere.float()


# ----------------------------------------------------------------------------- Stage 2
def sinusoid(x, dim, min_period, max_period):
    """x (...) -> (..., dim) with geometrically spaced periods in [min_period, max_period]."""
    half = dim // 2
    periods = torch.exp(torch.linspace(math.log(max_period), math.log(min_period), half,
                                       device=x.device))
    ang = x.float().unsqueeze(-1) * (2 * math.pi / periods)
    return torch.cat([ang.sin(), ang.cos()], dim=-1)


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


def _mamba(dim, d_state, d_conv, expand):
    """A standard (single-direction) Mamba layer from the bundled backend (the module used by
    MAMBA-DSR, already imported by the Jean-Zay env check); bimamba_type other than "v4" runs the
    plain selective scan, so its unused second-branch parameters are dropped."""
    from mamba_backend.bimamba_ssm.modules.mamba_simple import Mamba
    m = Mamba(dim, d_state=d_state, d_conv=d_conv, expand=expand, use_fast_path=True, bimamba_type="none")
    for name in ("A_b_log", "conv1d_b", "x_proj_b", "dt_proj_b", "D_b"):
        if hasattr(m, name):
            delattr(m, name)
    return m


class TemporalMixer(nn.Module):
    """Mixing along time for x (B*K, T, C) = the T frames of each of the K tokens.

    mamba: a forward and a time-reversed selective scan (two standard Mamba layers, summed).
           The K token sequences of a sample are laid end to end (B, K*T, C), because the fused
           scan kernel parallelises along the sequence and is ~6x slower on many length-T
           sequences; a little state carries over at token boundaries, which the per-frame tau
           embedding marks. (The bundled bimamba "v4" is not used: its second scan runs over
           reversed feature channels, i.e. it is forward-only in time.)
    attn:  self-attention over the T frames of each token (ablation)."""

    def __init__(self, dim, heads, kind="mamba", d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.kind = kind
        if kind == "mamba":
            self.fwd = _mamba(dim, d_state, d_conv, expand)
            self.bwd = _mamba(dim, d_state, d_conv, expand)
        elif kind == "attn":
            self.mixer = MHA(dim, heads)
        else:
            raise ValueError(f"unknown temporal mixer {kind!r} (mamba|attn)")

    def forward(self, x, k):
        if self.kind == "attn":
            return self.mixer(x)
        bk, t, c = x.shape
        seq = x.reshape(bk // k, k * t, c)
        return (self.fwd(seq) + self.bwd(seq.flip(1)).flip(1)).reshape(bk, t, c)


class SpaceTimeBlock(nn.Module):
    def __init__(self, dim, heads, cond_dim, temporal="mamba", dropout=0.0):
        super().__init__()
        self.n1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.space = MHA(dim, heads)
        self.n2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.time = TemporalMixer(dim, heads, temporal)
        self.n3 = nn.LayerNorm(dim, elementwise_affine=False)
        self.mlp = mlp(dim, dropout=dropout)
        self.drop = nn.Dropout(dropout)
        self.mod = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 9 * dim))
        nn.init.zeros_(self.mod[1].weight)   # adaLN-Zero: every block starts as the identity
        nn.init.zeros_(self.mod[1].bias)

    def forward(self, x, cond):
        b, t, k, c = x.shape
        s1, b1, g1, s2, b2, g2, s3, b3, g3 = self.mod(cond)[:, None, None].chunk(9, dim=-1)
        h = modulate(self.n1(x), b1, s1).reshape(b * t, k, c)
        x = x + g1 * self.drop(self.space(h)).view(b, t, k, c)
        h = modulate(self.n2(x), b2, s2).permute(0, 2, 1, 3).reshape(b * k, t, c)
        x = x + g2 * self.drop(self.time(h, k)).view(b, k, t, c).permute(0, 2, 1, 3)
        return x + g3 * self.mlp(modulate(self.n3(x), b3, s3))


class SpaceTimeFlow(nn.Module):
    """v(x_u, u | context): x_u holds noisy residual tokens at target frames.

    Inputs per frame: [state, reference] where state = clean latent (context) or the noisy
    residual (target) and reference = the latent baseline at that frame (context: itself;
    target: linear interpolation of the context latents, = the last context latent past the
    end). Per-frame embeddings: time position tau (in units of the context spacing) and a
    context/target flag; per-token: anchor position. Global condition: flow time u and
    log(raw frames per tau unit)."""

    def __init__(self, latent_dim=32, width=384, heads=8, blocks=12, temporal="mamba",
                 dropout=0.0, n_freq=8, grad_checkpoint=False):
        super().__init__()
        self.config = dict(latent_dim=latent_dim, width=width, heads=heads, blocks=blocks,
                           temporal=temporal, dropout=dropout, n_freq=n_freq)
        self.grad_checkpoint = grad_checkpoint
        self.fourier = Fourier(n_freq)
        self.inp = nn.Linear(2 * latent_dim, width)
        self.anchor_pe = nn.Linear(self.fourier.out_dim, width)
        self.tau_pe = nn.Sequential(nn.Linear(64, width), nn.SiLU(), nn.Linear(width, width))
        self.flag = nn.Embedding(2, width)
        self.u_emb = nn.Sequential(nn.Linear(256, width), nn.SiLU(), nn.Linear(width, width))
        self.dt_emb = nn.Sequential(nn.Linear(32, width), nn.SiLU(), nn.Linear(width, width))
        self.blocks = nn.ModuleList([SpaceTimeBlock(width, heads, width, temporal, dropout)
                                     for _ in range(blocks)])
        self.n_out = nn.LayerNorm(width, elementwise_affine=False)
        self.mod_out = nn.Sequential(nn.SiLU(), nn.Linear(width, 2 * width))
        self.out = nn.Linear(width, latent_dim)
        for layer in (self.mod_out[1], self.out):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, state, ref, anchors, tau, is_target, u, log_dt):
        """state/ref (B, T, K, d); anchors (B, K, 3); tau (B, T); is_target (B, T) bool;
        u (B,) flow time in [0, 1]; log_dt (B,) -> velocity (B, T, K, d)."""
        x = (self.inp(torch.cat([state, ref], dim=-1))
             + self.anchor_pe(self.fourier(anchors))[:, None]
             + self.tau_pe(sinusoid(tau, 64, 0.2, 80.0))[:, :, None]
             + self.flag(is_target.long())[:, :, None])
        cond = (self.u_emb(sinusoid(u * 1000.0, 256, 1.0, 10000.0))
                + self.dt_emb(sinusoid(log_dt, 32, 0.5, 50.0)))
        for blk in self.blocks:
            if self.grad_checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(blk, x, cond, use_reentrant=False)
            else:
                x = blk(x, cond)
        shift, scale = self.mod_out(cond)[:, None, None].chunk(2, dim=-1)
        return self.out(modulate(self.n_out(x), shift, scale))


def count_params(module):
    return sum(p.numel() for p in module.parameters())
