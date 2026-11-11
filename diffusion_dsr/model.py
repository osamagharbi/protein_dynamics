"""
Stage B denoiser: a small MLP that predicts noise on a per-frame latent
vector (dim = Stage A's --latent-size, e.g. 32), conditioned on the
diffusion timestep k and DSR's physical trajectory time t.

Much simpler than a point-cloud denoiser (no PointNet/permutation-invariance
needed) since we're now diffusing a compact latent, not raw points.
"""
import math

import torch
import torch.nn as nn


class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        # x: (B,) float tensor
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=x.device).float() / half)
        args = x.float().unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb


class ResBlock(nn.Module):
    def __init__(self, hidden, cond_dim):
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.fc1 = nn.Linear(hidden, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.cond_proj = nn.Linear(cond_dim, hidden)
        self.act = nn.SiLU()

    def forward(self, h, c):
        h_in = h
        h = self.norm(h)
        h = self.act(self.fc1(h) + self.cond_proj(c))
        h = self.fc2(h)
        return h_in + h


class LatentDenoiser(nn.Module):
    def __init__(self, latent_dim=32, k_emb_dim=128, t_emb_dim=32, hidden=256, n_blocks=4,
                 num_datasets=1, dataset_emb_dim=16):
        super().__init__()
        self.latent_dim = latent_dim

        self.k_embed = SinusoidalEmbedding(k_emb_dim)
        self.k_mlp = nn.Sequential(nn.Linear(k_emb_dim, k_emb_dim), nn.SiLU(), nn.Linear(k_emb_dim, k_emb_dim))

        self.t_embed = SinusoidalEmbedding(t_emb_dim)
        self.t_mlp = nn.Sequential(nn.Linear(t_emb_dim, 64), nn.SiLU(), nn.Linear(64, 64))

        # Coarse dataset-id conditioning, unused (num_datasets=1) until the
        # multi-dataset extension -- kept here so that step doesn't require
        # touching this file again.
        self.dataset_emb = nn.Embedding(num_datasets, dataset_emb_dim) if num_datasets > 1 else None
        cond_dim = k_emb_dim + 64 + (dataset_emb_dim if num_datasets > 1 else 0)

        self.in_proj = nn.Linear(latent_dim, hidden)
        self.cond_proj = nn.Linear(cond_dim, hidden)
        self.blocks = nn.ModuleList([ResBlock(hidden, cond_dim) for _ in range(n_blocks)])
        self.out_norm = nn.LayerNorm(hidden)
        self.out = nn.Linear(hidden, latent_dim)

    def forward(self, z_noisy, k, t, dataset_id=None):
        c = torch.cat([self.k_mlp(self.k_embed(k)), self.t_mlp(self.t_embed(t))], dim=-1)
        if self.dataset_emb is not None and dataset_id is not None:
            c = torch.cat([c, self.dataset_emb(dataset_id)], dim=-1)

        h = self.in_proj(z_noisy) + self.cond_proj(c)
        for block in self.blocks:
            h = block(h, c)
        h = self.out_norm(h)
        return self.out(h)


class MeanPredictor(nn.Module):
    """v2 hybrid head: deterministic map physical time t -> conditional-mean
    latent mu(t). Because Stage A v2's latent trajectory is now smooth in t,
    this is a near-perfect regression, so mu(t) alone nearly reconstructs each
    frame -- the diffusion residual then only has to model the small spread
    around it. Small MLP over a sinusoidal time embedding."""

    def __init__(self, latent_dim=8, t_emb_dim=64, hidden=128, n_blocks=3):
        super().__init__()
        self.latent_dim = latent_dim
        self.t_embed = SinusoidalEmbedding(t_emb_dim)
        self.in_proj = nn.Linear(t_emb_dim, hidden)
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.SiLU(),
                          nn.Linear(hidden, hidden))
            for _ in range(n_blocks)
        ])
        self.out_norm = nn.LayerNorm(hidden)
        self.out = nn.Linear(hidden, latent_dim)

    def forward(self, t):
        h = self.in_proj(self.t_embed(t))
        for block in self.blocks:
            h = h + block(h)
        return self.out(self.out_norm(h))


def count_params(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    m = LatentDenoiser(latent_dim=8)
    print(f"LatentDenoiser params: {count_params(m):,}")
    z = torch.randn(8, 8)
    k = torch.randint(0, 1000, (8,))
    t = torch.rand(8) * 2 - 1
    print("denoiser output shape:", m(z, k, t).shape)

    mp = MeanPredictor(latent_dim=8)
    print(f"MeanPredictor params: {count_params(mp):,}")
    print("mean output shape:", mp(t).shape)
