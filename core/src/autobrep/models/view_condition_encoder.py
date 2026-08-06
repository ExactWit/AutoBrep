"""Multi-view (3 RGB renders) + 3-view geometric TechDraw (DXF+SVG) → soft AR prefix."""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18

from autobrep.data.techdraw_dxf.schema import (
    GEOM_DIM,
    MAX_PRIMS_PER_VIEW,
    NUM_LINETYPES,
    NUM_PRIM_TYPES,
    NUM_TD_VIEWS,
)
from autobrep.models.prim_transformer_encoder import (
    PrimTransformerEncoder,
    SoftPrefixCompressor,
    TopoSketchEncoder,
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class TechDrawSetEncoder(nn.Module):
    """Set-Transformer over DXF/SVG primitives for one TechDraw view (P0 path)."""

    def __init__(
        self,
        out_dim: int = 256,
        d_model: int = 128,
        n_layers: int = 2,
        n_heads: int = 4,
        max_prims: int = MAX_PRIMS_PER_VIEW,
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
        multi = prim_types.ndim == 3
        if multi:
            b, v, n = prim_types.shape
            prim_types = prim_types.reshape(b * v, n)
            prim_linetypes = prim_linetypes.reshape(b * v, n)
            prim_geom = prim_geom.reshape(b * v, n, -1)
            prim_mask = prim_mask.reshape(b * v, n)

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
        out = self.out_proj(pooled)
        if multi:
            out = out.reshape(b, v, -1)
        return out


class ViewConditionEncoder(nn.Module):
    """
    Encode 3 RGB renders + TechDraw into M soft prefix embeddings for AR.

    Modes:
      - ``use_prim_seq_encoder=False`` (P0): per-view SetEncoder → 3 TD tokens
      - ``use_prim_seq_encoder=True`` + ``prim_prefix_mode=compress`` (P1-A):
        PrimTransformerEncoder → SoftPrefixCompressor (M=64) + image fuse
      - ``prim_prefix_mode=direct``: prim tokens truncated/padded to M (legacy)
      - ``prim_prefix_mode=prefix_lm``: image + all prim tokens as AR soft
        prefix; caller applies prefix-LM attn mask (CAD↛condition)
    """

    def __init__(
        self,
        dim: int = 2048,
        hidden: int = 256,
        num_latents: int = 64,
        num_image_views: int = 3,
        num_td_views: int = NUM_TD_VIEWS,
        num_heads: int = 4,
        dropout: float = 0.1,
        view_dropout_max: int = 1,
        pretrained_backbone: bool = True,
        use_prim_seq_encoder: bool = False,
        prim_d_model: int = 512,
        prim_n_layers: int = 4,
        prim_max_seq: int = 384,
        prim_geom_bins: int = 1024,
        prim_prefix_mode: str = "compress",
        use_topo_sketch: bool = False,
        topo_sketch_max: int = 64,
        cond_dropout: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.hidden = hidden
        self.num_latents = num_latents
        self.num_image_views = num_image_views
        self.num_td_views = num_td_views
        self.view_dropout_max = view_dropout_max
        self.use_prim_seq_encoder = bool(use_prim_seq_encoder)
        self.cond_dropout = float(cond_dropout)
        self._last_cond_drop_mask: torch.Tensor | None = None
        self.prim_prefix_mode = str(prim_prefix_mode)
        if self.prim_prefix_mode not in ("compress", "direct", "prefix_lm"):
            raise ValueError(f"unknown prim_prefix_mode: {prim_prefix_mode}")
        self.use_topo_sketch = bool(use_topo_sketch)
        self.topo_sketch_max = int(topo_sketch_max)
        self._last_prefix_valid: torch.Tensor | None = None

        weights = ResNet18_Weights.DEFAULT if pretrained_backbone else None
        backbone = resnet18(weights=weights)
        backbone.fc = nn.Identity()
        self.backbone = backbone  # → 512-d
        self.img_proj = nn.Sequential(
            nn.Linear(512, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )

        if self.use_prim_seq_encoder and self.prim_prefix_mode == "prefix_lm":
            # Equal-status condition tokens: embed (+MLP) → AR dim; no compressor /
            # external Transformer. Abstraction happens inside the shared AR stack.
            self.prim_encoder = PrimTransformerEncoder(
                d_model=dim,
                n_layers=0,
                n_heads=max(4, dim // 64),
                max_seq=prim_max_seq,
                geom_bins=prim_geom_bins,
                num_views=num_td_views,
                dropout=dropout,
                out_dim=dim,
            )
            self.prim_mlp = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim, dim),
            )
            self.img_to_ar = nn.Sequential(
                nn.Linear(512, dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim, dim),
            )
            self.prim_compressor = None
            self.fuse_latents = None
            self.fuse_cross_attn = None
            self.fuse_ff = None
            self.techdraw_encoder = None
            self.td_view_embed = None
            self.techdraw_token = None
            self.latents = None
            self.cross_attn = None
            self.ff = None
        elif self.use_prim_seq_encoder:
            # Align prim encoder out_dim with hidden so img+prim share mem space.
            self.prim_encoder = PrimTransformerEncoder(
                d_model=prim_d_model,
                n_layers=prim_n_layers,
                n_heads=max(4, prim_d_model // 64),
                max_seq=prim_max_seq,
                geom_bins=prim_geom_bins,
                num_views=num_td_views,
                dropout=dropout,
                out_dim=hidden,
            )
            self.prim_mlp = None
            self.img_to_ar = None
            self.prim_compressor = SoftPrefixCompressor(
                d_model=hidden,
                num_latents=num_latents,
                n_heads=num_heads,
                dropout=dropout,
            )
            # Extra cross-attn fuse: latents attend to [img | compressed prim latents]
            # Already compressed to M; we still fuse with image tokens via shared latents.
            self.fuse_latents = nn.Parameter(torch.randn(num_latents, hidden) * 0.02)
            self.fuse_cross_attn = nn.MultiheadAttention(
                embed_dim=hidden,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.fuse_ff = nn.Sequential(
                nn.LayerNorm(hidden),
                nn.Linear(hidden, hidden * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden * 4, hidden),
            )
            self.techdraw_encoder = None
            self.td_view_embed = None
            self.techdraw_token = None
            self.latents = None
            self.cross_attn = None
            self.ff = None
        else:
            self.prim_encoder = None
            self.prim_mlp = None
            self.img_to_ar = None
            self.prim_compressor = None
            self.fuse_latents = None
            self.fuse_cross_attn = None
            self.fuse_ff = None
            self.techdraw_encoder = TechDrawSetEncoder(out_dim=hidden, dropout=dropout)
            self.td_view_embed = nn.Embedding(num_td_views, hidden)
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

        # Topology sketch branch (loop-group → coarse skeleton scaffold tokens).
        if self.use_topo_sketch:
            self.topo_encoder = TopoSketchEncoder(
                d_model=hidden,
                out_dim=dim,
                max_sketch=topo_sketch_max,
                num_views=num_td_views,
                dropout=dropout,
            )
            # Aux cardinality head: predict log-count of faces & edges from sketch.
            self.topo_count_head = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim // 2),
                nn.GELU(),
                nn.Linear(dim // 2, 2),
            )
        else:
            self.topo_encoder = None
            self.topo_count_head = None

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

    def _encode_images(self, images: torch.Tensor) -> torch.Tensor:
        if images.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            images = images.float()
        images = self._modality_dropout(images)
        images = (images - self.img_mean.to(dtype=images.dtype)) / self.img_std.to(
            dtype=images.dtype
        )
        b, v, c, h, w = images.shape
        assert v == self.num_image_views, f"expected {self.num_image_views} image views, got {v}"
        flat = images.reshape(b * v, c, h, w)
        feats = self.backbone(flat.float()).to(dtype=images.dtype)
        if self.img_to_ar is not None:
            # prefix_lm: map ResNet feats straight to AR dim
            return self.img_to_ar(feats).reshape(b, v, -1), images
        return self.img_proj(feats).reshape(b, v, -1), images

    def forward(
        self,
        images: torch.Tensor,
        prim_types: torch.Tensor,
        prim_linetypes: torch.Tensor,
        prim_geom: torch.Tensor,
        prim_mask: torch.Tensor,
        *,
        prim_group_roles: torch.Tensor | None = None,
        prim_group_ids: torch.Tensor | None = None,
        drop_techdraw: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            images: (B, 3, 3, H, W) float in [0, 1]
            prim_*: (B, V, N) / (B, V, N, G)
            prim_group_roles: optional (B, V, N); defaults to 0
            prim_group_ids: optional (B, V, N); >0 marks loop membership
        Returns:
            prepend_embeds: (B, [S+]M+2, dim); topo sketch prepended when enabled
        """
        img_tokens, images = self._encode_images(images)
        b = images.shape[0]
        dtype = images.dtype

        # Legacy flat (B, N) → pad to V views
        if prim_types.ndim == 2:
            prim_types = prim_types.unsqueeze(1).expand(-1, self.num_td_views, -1)
            prim_linetypes = prim_linetypes.unsqueeze(1).expand(-1, self.num_td_views, -1)
            prim_geom = prim_geom.unsqueeze(1).expand(-1, self.num_td_views, -1, -1)
            prim_mask = prim_mask.unsqueeze(1)
            if prim_group_roles is not None:
                prim_group_roles = prim_group_roles.unsqueeze(1)
            if self.num_td_views > 1:
                prim_mask = prim_mask.repeat(1, self.num_td_views, 1)
                prim_mask[:, 1:] = False
                if prim_group_roles is not None:
                    prim_group_roles = prim_group_roles.repeat(1, self.num_td_views, 1)
                    prim_group_roles[:, 1:] = 0
                if prim_group_ids is not None:
                    prim_group_ids = prim_group_ids.repeat(1, self.num_td_views, 1)
                    prim_group_ids[:, 1:] = 0

        if self.use_prim_seq_encoder:
            prim_seq, prim_seq_mask = self.prim_encoder(
                prim_types,
                prim_linetypes,
                prim_geom.float(),
                prim_mask,
                prim_group_roles=prim_group_roles,
            )
            prim_seq = prim_seq.to(dtype=dtype)
            # Per-sample TechDraw condition dropout (classifier-free style):
            # dropped samples see an all-zero prim prefix while image latents stay.
            cond_drop = None
            if self.training and not drop_techdraw:
                if self.cond_dropout > 0:
                    cond_drop = (
                        torch.rand(b, device=images.device) < self.cond_dropout
                    )
            elif drop_techdraw:
                cond_drop = torch.ones(b, dtype=torch.bool, device=images.device)
            self._last_cond_drop_mask = cond_drop
            if cond_drop is not None and bool(cond_drop.any()):
                prim_seq = prim_seq.masked_fill(cond_drop.view(b, 1, 1), 0)
                prim_seq_mask = prim_seq_mask & ~cond_drop.view(b, 1)

            if self.prim_prefix_mode == "direct":
                # One token per primitive (no compressor); cap at num_latents.
                key_pad = ~prim_seq_mask
                L = prim_seq.shape[1]
                if L >= self.num_latents:
                    hidden = prim_seq[:, : self.num_latents]
                else:
                    pad = self.num_latents - L
                    hidden = torch.cat(
                        [prim_seq, torch.zeros(b, pad, prim_seq.shape[-1], dtype=dtype, device=prim_seq.device)],
                        dim=1,
                    )
                self._last_prefix_valid = None
            elif self.prim_prefix_mode == "prefix_lm":
                # Embed(+MLP) prims and image tokens already in AR dim; shared
                # AR stack + prefix-LM mask does the abstraction (equal status).
                if self.prim_mlp is not None:
                    prim_seq = self.prim_mlp(prim_seq)
                hidden = torch.cat([img_tokens, prim_seq], dim=1)  # (B, 3+L, dim)
                img_mask = torch.ones(
                    b, self.num_image_views, dtype=torch.bool, device=images.device
                )
                cond_token_mask = torch.cat([img_mask, prim_seq_mask], dim=1)
                self._last_prefix_valid = cond_token_mask
            else:
                # Compress prim seq → M; fuse with image tokens via another cross-attn
                key_pad = ~prim_seq_mask
                empty = ~prim_seq_mask.any(dim=1)
                if empty.any():
                    key_pad = key_pad.clone()
                    key_pad[empty, 0] = False
                prim_latents = self.prim_compressor(prim_seq, key_padding_mask=key_pad)
                mem = torch.cat([img_tokens, prim_latents], dim=1)  # (B, 3+M, H)
                q = self.fuse_latents.unsqueeze(0).expand(b, -1, -1)
                attn_out, _ = self.fuse_cross_attn(q, mem, mem, need_weights=False)
                hidden = q + attn_out
                hidden = hidden + self.fuse_ff(hidden)
                self._last_prefix_valid = None
        else:
            self._last_prefix_valid = None
            td = self.techdraw_encoder(
                prim_types, prim_linetypes, prim_geom.float(), prim_mask
            ).to(dtype=dtype)
            if self.training and not drop_techdraw:
                if torch.rand((), device=images.device) < 0.1:
                    td = torch.zeros_like(td)
            elif drop_techdraw:
                td = torch.zeros_like(td)
            view_ids = torch.arange(self.num_td_views, device=images.device).view(1, -1)
            td = td + self.td_view_embed(view_ids).to(dtype=td.dtype)
            td_tokens = self.techdraw_token(td)
            mem = torch.cat([img_tokens, td_tokens], dim=1)
            latents = self.latents.unsqueeze(0).expand(b, -1, -1)
            attn_out, _ = self.cross_attn(latents, mem, mem, need_weights=False)
            hidden = latents + attn_out
            hidden = hidden + self.ff(hidden)

        if self.prim_prefix_mode == "prefix_lm":
            # Already AR-dim; skip shared hidden→dim proj used by compress/direct.
            tokens = hidden
        else:
            tokens = self.proj(hidden)
        bos = self.bos_view.expand(b, -1, -1)
        eos = self.eos_view.expand(b, -1, -1)
        prefix = torch.cat([bos, tokens, eos], dim=1)

        # Wrap condition validity with always-valid BOS/EOS soft markers.
        if self._last_prefix_valid is not None:
            ones = torch.ones(b, 1, dtype=torch.bool, device=prefix.device)
            self._last_prefix_valid = torch.cat(
                [ones, self._last_prefix_valid.to(device=prefix.device), ones], dim=1
            )

        if self.use_topo_sketch and self.topo_encoder is not None:
            sketch, sketch_mask = self.topo_encoder(
                prim_types,
                prim_linetypes,
                prim_geom.float(),
                prim_mask,
                prim_group_ids=prim_group_ids,
                prim_group_roles=prim_group_roles,
            )
            sketch = sketch.to(dtype=dtype)
            # Keep topo sketch consistent with prim-condition dropout, else the
            # dropped condition leaks back in through the sketch tokens.
            cond_drop = self._last_cond_drop_mask
            if cond_drop is not None and bool(cond_drop.any()):
                sketch = sketch.masked_fill(cond_drop.view(b, 1, 1), 0)
                sketch_mask = sketch_mask & ~cond_drop.view(b, 1)
            self._last_topo_sketch = (sketch, sketch_mask)
            # Topology scaffold goes FIRST so AR attends it before geometry.
            prefix = torch.cat([sketch, prefix], dim=1)
            if self._last_prefix_valid is not None:
                # Sketch tokens are condition-side; treat pad sketch as invalid.
                self._last_prefix_valid = torch.cat(
                    [sketch_mask.to(dtype=torch.bool), self._last_prefix_valid], dim=1
                )
        else:
            self._last_topo_sketch = None
        return prefix

    def get_topo_sketch(self):
        """Return (sketch, mask) from the last forward (keeps grad for aux loss)."""
        return self._last_topo_sketch
