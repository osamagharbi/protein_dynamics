"""Bidirectional-Mamba latent predictor for MAMBA-DSR.

Stage B is deliberately a direct regressor, not a diffusion model. The
latent dimensions are treated as an ordered sequence of scalar channels. A
shared physical-time embedding conditions every token, and bidirectional
Mamba blocks model interactions between latent channels before projecting the
sequence back to one latent vector.
"""
import math

import torch
import torch.nn as nn

from mamba_backend.bimamba_ssm.modules.mamba_simple import Mamba


class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=x.device).float() / half
        )
        args = x.float().unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb


class BidirectionalMambaBlock(nn.Module):
    """Pre-norm residual block around the bundled bidirectional Mamba."""

    def __init__(self, hidden, d_state=16, d_conv=4, expand=2,
                 bimamba_type="v4"):
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.mixer = Mamba(
            hidden,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            bimamba_type=bimamba_type,
            use_fast_path=True,
        )

    def forward(self, x):
        return x + self.mixer(self.norm(x))


class MambaLatentPredictor(nn.Module):
    """Predict the complete DSR latent directly from normalized physical time."""

    def __init__(self, latent_dim=8, time_emb_dim=64, hidden=64, depth=3,
                 d_state=16, d_conv=4, expand=2, bimamba_type="v4"):
        super().__init__()
        self.latent_dim = latent_dim
        self.model_config = {
            "latent_dim": latent_dim,
            "time_emb_dim": time_emb_dim,
            "hidden": hidden,
            "depth": depth,
            "d_state": d_state,
            "d_conv": d_conv,
            "expand": expand,
            "bimamba_type": bimamba_type,
        }

        self.time_embed = SinusoidalEmbedding(time_emb_dim)
        self.time_proj = nn.Sequential(
            nn.Linear(time_emb_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )
        self.channel_tokens = nn.Parameter(torch.randn(latent_dim, hidden) * 0.02)
        self.blocks = nn.ModuleList([
            BidirectionalMambaBlock(
                hidden, d_state=d_state, d_conv=d_conv, expand=expand,
                bimamba_type=bimamba_type,
            )
            for _ in range(depth)
        ])
        self.out_norm = nn.LayerNorm(hidden)
        self.out = nn.Linear(hidden, 1)

    def forward(self, t):
        # B,L,H: one learned token per latent channel plus shared time context.
        h = self.channel_tokens.unsqueeze(0) + self.time_proj(
            self.time_embed(t)
        ).unsqueeze(1)
        for block in self.blocks:
            h = block(h)
        return self.out(self.out_norm(h)).squeeze(-1)


def count_params(model):
    return sum(parameter.numel() for parameter in model.parameters())


if __name__ == "__main__":
    model = MambaLatentPredictor(latent_dim=8)
    t = torch.rand(4)
    print(f"MambaLatentPredictor params: {count_params(model):,}")
    print("output shape:", model(t).shape)
