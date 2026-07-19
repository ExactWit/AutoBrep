"""Lightweight point-cloud → soft prefix tokens for frozen AutoBrep AR."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PointCloudConditionEncoder(nn.Module):
    """
    Encode an unordered point cloud (B, N, 3) into M continuous prefix
    embeddings of size `dim`, consumed by XTransformer via `prepend_embeds`.
    """

    def __init__(
        self,
        dim: int = 2048,
        hidden: int = 256,
        num_latents: int = 64,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.num_latents = num_latents

        self.point_mlp = nn.Sequential(
            nn.Linear(3, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.latents = nn.Parameter(torch.randn(num_latents, hidden) * 0.02)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ff = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 4, hidden),
        )
        self.proj = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        # Boundary markers (learnable) so the AR sees explicit PC block edges.
        self.bos_pc = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.eos_pc = nn.Parameter(torch.randn(1, 1, dim) * 0.02)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """
        Args:
            points: (B, N, 3) float tensor in roughly [-1, 1]
        Returns:
            prepend_embeds: (B, M+2, dim) including BOPC/EOPC soft markers
        """
        b = points.shape[0]
        x = self.point_mlp(points)  # (B, N, H)
        latents = self.latents.unsqueeze(0).expand(b, -1, -1)
        attn_out, _ = self.cross_attn(latents, x, x, need_weights=False)
        h = latents + attn_out
        h = h + self.ff(h)
        tokens = self.proj(h)  # (B, M, D)
        bos = self.bos_pc.expand(b, -1, -1)
        eos = self.eos_pc.expand(b, -1, -1)
        return torch.cat([bos, tokens, eos], dim=1)


def normalize_point_cloud(points: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Per-cloud center + scale to fit [-1, 1] (batch)."""
    mins = points.amin(dim=1, keepdim=True)
    maxs = points.amax(dim=1, keepdim=True)
    center = 0.5 * (mins + maxs)
    centered = points - center
    scale = centered.abs().amax(dim=(1, 2), keepdim=True).clamp_min(eps)
    return centered / scale
