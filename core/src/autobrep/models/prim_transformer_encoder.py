"""Primitive-sequence Transformer encoder for TechDraw soft prefixes (P1-A)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from autobrep.data.techdraw_dxf.schema import (
    GEOM_DIM,
    NUM_GROUP_ROLES,
    NUM_LINETYPES,
    NUM_PRIM_TYPES,
    NUM_TD_VIEWS,
)


def quantize_geom(geom: torch.Tensor, *, n_bins: int = 1024) -> torch.Tensor:
    """
    Map continuous geom (typically in [-1, 1]) to bin indices in ``[0, n_bins)``.

    Values outside [-1, 1] are clamped.
    """
    x = geom.float().clamp(-1.0, 1.0)
    # [-1,1] → [0, n_bins-1]
    idx = ((x + 1.0) * 0.5 * (n_bins - 1)).round().long()
    return idx.clamp(0, n_bins - 1)


class PrimTransformerEncoder(nn.Module):
    """
    Encode up to ``max_seq`` TechDraw primitives across views into a token sequence.

    Token = type + 1024-bin geom (sum over dims) + linetype + group_role + view + pos.
    """

    def __init__(
        self,
        d_model: int = 512,
        n_layers: int = 4,
        n_heads: int = 8,
        max_seq: int = 384,
        geom_bins: int = 1024,
        geom_dim: int = GEOM_DIM,
        num_views: int = NUM_TD_VIEWS,
        dropout: float = 0.1,
        out_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_seq = max_seq
        self.geom_bins = geom_bins
        self.geom_dim = geom_dim
        self.num_views = num_views
        self.out_dim = out_dim or d_model

        self.type_embed = nn.Embedding(NUM_PRIM_TYPES, d_model)
        self.linetype_embed = nn.Embedding(NUM_LINETYPES, d_model)
        self.group_role_embed = nn.Embedding(NUM_GROUP_ROLES, d_model)
        self.view_embed = nn.Embedding(num_views, d_model)
        self.geom_bin_embed = nn.Embedding(geom_bins, d_model)
        self.geom_dim_embed = nn.Embedding(geom_dim, d_model)
        self.pos_embed = nn.Embedding(max_seq, d_model)
        self.drop = nn.Dropout(dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.out_proj = (
            nn.Identity()
            if self.out_dim == d_model
            else nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, self.out_dim))
        )

    def _flatten_views(
        self,
        prim_types: torch.Tensor,
        prim_linetypes: torch.Tensor,
        prim_geom: torch.Tensor,
        prim_mask: torch.Tensor,
        prim_group_roles: torch.Tensor | None,
    ) -> tuple[torch.Tensor, ...]:
        """(B,V,N,*) → packed (B, L≤max_seq, *) with view ids."""
        if prim_types.ndim == 2:
            # (B,N) → (B,1,N)
            prim_types = prim_types.unsqueeze(1)
            prim_linetypes = prim_linetypes.unsqueeze(1)
            prim_geom = prim_geom.unsqueeze(1)
            prim_mask = prim_mask.unsqueeze(1)
            if prim_group_roles is not None:
                prim_group_roles = prim_group_roles.unsqueeze(1)

        b, v, n = prim_types.shape
        device = prim_types.device
        view_ids = (
            torch.arange(v, device=device)
            .view(1, v, 1)
            .expand(b, v, n)
            .reshape(b, v * n)
        )
        types = prim_types.reshape(b, v * n)
        lts = prim_linetypes.reshape(b, v * n)
        geom = prim_geom.reshape(b, v * n, -1)
        mask = prim_mask.reshape(b, v * n)
        if prim_group_roles is None:
            roles = torch.zeros_like(types)
        else:
            roles = prim_group_roles.reshape(b, v * n)

        # Keep valid tokens first, truncate to max_seq
        L = min(self.max_seq, v * n)
        out_types = torch.zeros(b, L, dtype=types.dtype, device=device)
        out_lts = torch.zeros(b, L, dtype=lts.dtype, device=device)
        out_roles = torch.zeros(b, L, dtype=roles.dtype, device=device)
        out_views = torch.zeros(b, L, dtype=view_ids.dtype, device=device)
        out_geom = torch.zeros(b, L, geom.shape[-1], dtype=geom.dtype, device=device)
        out_mask = torch.zeros(b, L, dtype=torch.bool, device=device)

        for i in range(b):
            valid = mask[i].nonzero(as_tuple=False).flatten()
            if valid.numel() == 0:
                continue
            take = valid[:L]
            m = take.numel()
            out_types[i, :m] = types[i, take]
            out_lts[i, :m] = lts[i, take]
            out_roles[i, :m] = roles[i, take]
            out_views[i, :m] = view_ids[i, take]
            out_geom[i, :m] = geom[i, take]
            out_mask[i, :m] = True
        return out_types, out_lts, out_roles, out_views, out_geom, out_mask

    def forward(
        self,
        prim_types: torch.Tensor,
        prim_linetypes: torch.Tensor,
        prim_geom: torch.Tensor,
        prim_mask: torch.Tensor,
        prim_group_roles: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            tokens: (B, L, out_dim)
            mask: (B, L) True = valid
        """
        types, lts, roles, views, geom, mask = self._flatten_views(
            prim_types, prim_linetypes, prim_geom, prim_mask, prim_group_roles
        )
        b, L = types.shape
        bins = quantize_geom(geom, n_bins=self.geom_bins)  # (B,L,G)
        # sum geom bin embeds + dim embeds
        g_emb = self.geom_bin_embed(bins)  # (B,L,G,D)
        dim_ids = torch.arange(self.geom_dim, device=geom.device).view(1, 1, -1)
        g_emb = g_emb + self.geom_dim_embed(dim_ids)
        g_emb = g_emb.mean(dim=2)

        pos = torch.arange(L, device=types.device).unsqueeze(0).expand(b, -1)
        tokens = (
            self.type_embed(types.clamp(0, NUM_PRIM_TYPES - 1))
            + self.linetype_embed(lts.clamp(0, NUM_LINETYPES - 1))
            + self.group_role_embed(roles.clamp(0, NUM_GROUP_ROLES - 1))
            + self.view_embed(views.clamp(0, self.num_views - 1))
            + g_emb
            + self.pos_embed(pos)
        )
        tokens = self.drop(tokens)

        pad = ~mask
        empty = ~mask.any(dim=1)
        if empty.any():
            pad = pad.clone()
            pad[empty, 0] = False
            tokens = tokens.clone()
            tokens[empty, 0] = 0.0

        hidden = self.encoder(tokens, src_key_padding_mask=pad)
        hidden = self.out_proj(hidden)
        if empty.any():
            hidden = hidden.clone()
            hidden[empty] = 0.0
        return hidden, mask


class SoftPrefixCompressor(nn.Module):
    """One cross-attn layer: learnable latents (M) attend over prim sequence (+ optional mem)."""

    def __init__(
        self,
        d_model: int = 512,
        num_latents: int = 64,
        n_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, d_model) * 0.02)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ff = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(
        self, mem: torch.Tensor, key_padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        b = mem.shape[0]
        q = self.latents.unsqueeze(0).expand(b, -1, -1)
        # key_padding_mask: True = ignore
        out, _ = self.cross_attn(
            q, mem, mem, key_padding_mask=key_padding_mask, need_weights=False
        )
        h = q + out
        return h + self.ff(h)


class TopoSketchEncoder(nn.Module):
    """
    Aggregate TechDraw loop groups into topology-sketch tokens (one per loop).

    Input is the raw per-view primitive set (B, V, N, ...). Primitives sharing a
    positive ``prim_group_id`` (and the same view) belong to the same loop /
    contour. Isolated primitives (group_id == 0, role == isolated) are emitted
    as singleton tokens so the sketch length reflects the topological entity
    count (loops + isolated contours ≈ face/edge skeleton cardinality).

    Each sketch token = role_embed + view_embed + MLP(type_hist ∥ log_size ∥
    closed_frac). No cross-attention: pure mean-pool aggregation so it cannot
    perturb the frozen AR prior beyond providing a coarse scaffold.
    """

    def __init__(
        self,
        d_model: int = 512,
        out_dim: int | None = None,
        max_sketch: int = 64,
        num_views: int = NUM_TD_VIEWS,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.out_dim = out_dim or d_model
        self.max_sketch = max_sketch
        self.num_views = num_views

        self.role_embed = nn.Embedding(NUM_GROUP_ROLES, d_model)
        self.view_embed = nn.Embedding(num_views, d_model)
        feat_dim = NUM_PRIM_TYPES + 2  # type histogram ∥ log size ∥ closed frac
        self.feat_proj = nn.Sequential(
            nn.Linear(feat_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.drop = nn.Dropout(dropout)
        self.out_proj = (
            nn.Identity()
            if self.out_dim == d_model
            else nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, self.out_dim))
        )

    def forward(
        self,
        prim_types: torch.Tensor,
        prim_linetypes: torch.Tensor,
        prim_geom: torch.Tensor,
        prim_mask: torch.Tensor,
        prim_group_ids: torch.Tensor | None = None,
        prim_group_roles: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            prim_*: (B, V, N) / (B, V, N, G)
            prim_group_ids: (B, V, N) int; >0 marks loop membership
        Returns:
            tokens: (B, S≤max_sketch, out_dim)
            mask: (B, S) True = valid sketch token
        """
        if prim_types.ndim == 2:
            prim_types = prim_types.unsqueeze(1)
            prim_linetypes = prim_linetypes.unsqueeze(1)
            prim_geom = prim_geom.unsqueeze(1)
            prim_mask = prim_mask.unsqueeze(1)
            if prim_group_ids is not None:
                prim_group_ids = prim_group_ids.unsqueeze(1)
            if prim_group_roles is not None:
                prim_group_roles = prim_group_roles.unsqueeze(1)

        b, v, n = prim_types.shape
        device = prim_types.device
        if prim_group_ids is None:
            prim_group_ids = torch.zeros_like(prim_types)
        if prim_group_roles is None:
            prim_group_roles = torch.zeros_like(prim_types)

        out_tokens = torch.zeros(b, self.max_sketch, self.d_model, device=device)
        out_mask = torch.zeros(b, self.max_sketch, dtype=torch.bool, device=device)

        type_eye = torch.eye(NUM_PRIM_TYPES, device=device)
        for i in range(b):
            slots: list[tuple[int, int, torch.Tensor]] = []  # (role, view, member idx)
            seen: dict[tuple[int, int], int] = {}
            for vi in range(v):
                valid = prim_mask[i, vi].nonzero(as_tuple=False).flatten()
                if valid.numel() == 0:
                    continue
                for idx in valid.tolist():
                    gid = int(prim_group_ids[i, vi, idx].item())
                    role = int(prim_group_roles[i, vi, idx].item())
                    if gid > 0:
                        key = (vi, gid)
                        if key not in seen:
                            seen[key] = len(slots)
                            slots.append((role, vi, []))
                        slots[seen[key]][2].append(idx)
                    else:
                        slots.append((role, vi, [idx]))
            if not slots:
                continue
            take = slots[: self.max_sketch]
            for s, (role, vi, members) in enumerate(take):
                mem = torch.tensor(members, device=device, dtype=torch.long)
                types = prim_types[i, vi, mem].clamp(0, NUM_PRIM_TYPES - 1)
                hist = type_eye[types].mean(dim=0)
                size = torch.log1p(torch.tensor(float(len(members)), device=device))
                closed = (
                    (prim_group_ids[i, vi, mem] > 0).float().mean()
                )
                feat = torch.cat([hist, size.view(1), closed.view(1)])
                tok = (
                    self.role_embed(torch.tensor(role, device=device).clamp(0, NUM_GROUP_ROLES - 1))
                    + self.view_embed(torch.tensor(vi, device=device).clamp(0, self.num_views - 1))
                    + self.feat_proj(feat)
                )
                out_tokens[i, s] = tok
                out_mask[i, s] = True

        out_tokens = self.drop(out_tokens)
        out_tokens = self.out_proj(out_tokens)
        empty = ~out_mask.any(dim=1)
        if empty.any():
            out_tokens = out_tokens.clone()
            out_tokens[empty] = 0.0
        return out_tokens, out_mask


__all__ = ["PrimTransformerEncoder", "SoftPrefixCompressor", "TopoSketchEncoder", "quantize_geom"]
