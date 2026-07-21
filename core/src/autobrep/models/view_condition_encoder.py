"""Multi-view (3 RGB renders) + structured TechDraw DXF → soft AR prefix tokens."""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18

from autobrep.data.techdraw_dxf.schema import GEOM_DIM, MAX_PRIMS, NUM_LINETYPES, NUM_PRIM_TYPES

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class TechDrawSetEncoder(nn.Module):
    """Set-Transformer over DXF primitives (structured TechDraw / 三视图)."""

    def __init__(
        self,
        out_dim: int = 256,
        d_model: int = 128,
        n_layers: int = 2,
        n_heads: int = 4,
        max_prims: int = MAX_PRIMS,
        geom_dim: int = GEOM_DIM,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.out_dim = out_dim
        self.max_prims = max_prims
        self.type_embed = nn.Embedding(NUM_PRIM_TYPES, d_model)
        self.linetype_embed = nn.Embedding(NUM_LINETYPES, d_model)
        self.geom_mlp = nn.Sequential(
            nn.Linear(geom_dim, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.out_proj = nn.Sequential(
            nn.Linear(d_model, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(
        self,
        prim_types: torch.Tensor,
        prim_linetypes: torch.Tensor,
        prim_geom: torch.Tensor,
        prim_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns:
            (B, out_dim) pooled TechDraw feature.
        """
        tokens = (
            self.type_embed(prim_types.clamp(min=0, max=NUM_PRIM_TYPES - 1))
            + self.linetype_embed(prim_linetypes.clamp(min=0, max=NUM_LINETYPES - 1))
            + self.geom_mlp(prim_geom)
        )
        padding_mask = ~prim_mask.bool()
        empty = ~prim_mask.any(dim=1)
        if empty.any():
            padding_mask = padding_mask.clone()
            padding_mask[empty, 0] = False
            tokens = tokens.clone()
            tokens[empty, 0] = 0.0

        hidden = self.encoder(tokens, src_key_padding_mask=padding_mask)
        weights = prim_mask.float().unsqueeze(-1)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp(min=1.0)
        pooled = (hidden * weights).sum(dim=1)
        pooled = torch.where(empty.unsqueeze(-1), torch.zeros_like(pooled), pooled)
        return self.out_proj(pooled)


class ViewConditionEncoder(nn.Module):
    """
    Encode 3 RGB renders + structured TechDraw DXF into M prefix embeddings.

    Images are NOT used for TechDraw — DXF primitives are set-encoded.
    """

    def __init__(
        self,
        dim: int = 2048,
        hidden: int = 256,
        num_latents: int = 64,
        num_image_views: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
        view_dropout_max: int = 1,
        pretrained_backbone: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.num_latents = num_latents
        self.num_image_views = num_image_views
        self.view_dropout_max = view_dropout_max

        weights = ResNet18_Weights.DEFAULT if pretrained_backbone else None
        backbone = resnet18(weights=weights)
        backbone.fc = nn.Identity()
        self.backbone = backbone  # → 512-d
        self.img_proj = nn.Sequential(
            nn.Linear(512, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.techdraw_encoder = TechDrawSetEncoder(out_dim=hidden, dropout=dropout)
        self.techdraw_token = nn.Sequential(
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
        self.bos_view = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.eos_view = nn.Parameter(torch.randn(1, 1, dim) * 0.02)

        mean = torch.tensor(IMAGENET_MEAN).view(1, 1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(1, 1, 3, 1, 1)
        self.register_buffer("img_mean", mean, persistent=False)
        self.register_buffer("img_std", std, persistent=False)

    def _modality_dropout(self, images: torch.Tensor) -> torch.Tensor:
        if not self.training or self.view_dropout_max <= 0:
            return images
        b, v = images.shape[:2]
        out = images.clone()
        for i in range(b):
            n_drop = int(torch.randint(0, self.view_dropout_max + 1, (1,)).item())
            if n_drop <= 0:
                continue
            idx = torch.randperm(v, device=images.device)[:n_drop]
            out[i, idx] = 0.0
        return out

    def forward(
        self,
        images: torch.Tensor,
        prim_types: torch.Tensor,
        prim_linetypes: torch.Tensor,
        prim_geom: torch.Tensor,
        prim_mask: torch.Tensor,
        *,
        drop_techdraw: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            images: (B, 3, 3, H, W) float in [0, 1] — render views only
            prim_*: TechDraw DXF tensors
        Returns:
            prepend_embeds: (B, M+2, dim)
        """
        if images.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            images = images.float()
        images = self._modality_dropout(images)
        images = (images - self.img_mean.to(dtype=images.dtype)) / self.img_std.to(
            dtype=images.dtype
        )

        b, v, c, h, w = images.shape
        assert v == self.num_image_views, f"expected {self.num_image_views} image views, got {v}"
        flat = images.reshape(b * v, c, h, w)
        # ResNet expects float32 for BN stability under amp callers.
        feats = self.backbone(flat.float()).to(dtype=images.dtype)
        img_tokens = self.img_proj(feats).reshape(b, v, -1)

        td = self.techdraw_encoder(
            prim_types, prim_linetypes, prim_geom.float(), prim_mask
        ).to(dtype=images.dtype)
        if self.training and drop_techdraw is False:
            # Occasional TechDraw dropout for robustness
            if torch.rand((), device=images.device) < 0.1:
                td = torch.zeros_like(td)
        elif drop_techdraw:
            td = torch.zeros_like(td)
        td_token = self.techdraw_token(td).unsqueeze(1)  # (B, 1, H)

        mem = torch.cat([img_tokens, td_token], dim=1)  # (B, 4, H)
        latents = self.latents.unsqueeze(0).expand(b, -1, -1)
        attn_out, _ = self.cross_attn(latents, mem, mem, need_weights=False)
        hidden = latents + attn_out
        hidden = hidden + self.ff(hidden)
        tokens = self.proj(hidden)
        bos = self.bos_view.expand(b, -1, -1)
        eos = self.eos_view.expand(b, -1, -1)
        return torch.cat([bos, tokens, eos], dim=1)
