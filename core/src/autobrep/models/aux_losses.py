"""P2 auxiliary losses (switchable): view bbox consistency + optional surf-type CE."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass
class AuxLossResult:
    total: torch.Tensor
    view_bbox: float
    surf_type: float

    def log_dict(self, prefix: str = "train/aux") -> dict[str, float]:
        return {
            f"{prefix}/view_bbox": self.view_bbox,
            f"{prefix}/surf_type": self.surf_type,
            f"{prefix}/total": float(self.total.detach().item())
            if self.total.numel() == 1
            else float(self.total.detach().mean().item()),
        }


def _view_aabb(
    prim_geom: torch.Tensor, prim_mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Approximate per-view AABB from line endpoints in geom slots 0:4.

    Args:
        prim_geom: (B, V, N, G)
        prim_mask: (B, V, N)
    Returns:
        mins, maxs: (B, V, 2)
    """
    # Use x0,y0,x1,y1 when present; else first 2 dims
    g = prim_geom.float()
    xy = torch.stack(
        [
            g[..., 0],
            g[..., 1],
            g[..., 2],
            g[..., 3],
        ],
        dim=-1,
    )  # (B,V,N,4)
    pts = xy.reshape(*xy.shape[:-1], 2, 2)  # (B,V,N,2,2)
    pts = pts.reshape(g.shape[0], g.shape[1], -1, 2)  # (B,V,2N,2)
    # Each prim → 2 endpoints; expand mask accordingly
    mask2 = prim_mask.unsqueeze(-1).expand(-1, -1, -1, 2).reshape(
        g.shape[0], g.shape[1], -1
    )  # (B,V,2N)
    mask2 = mask2.unsqueeze(-1).expand(-1, -1, -1, 2)  # (B,V,2N,2)
    big = torch.tensor(1e6, device=g.device, dtype=g.dtype)
    mins = pts.masked_fill(~mask2.bool(), big).amin(dim=2)
    maxs = pts.masked_fill(~mask2.bool(), -big).amax(dim=2)
    empty = ~prim_mask.any(dim=-1)
    mins = torch.where(empty.unsqueeze(-1), torch.zeros_like(mins), mins)
    maxs = torch.where(empty.unsqueeze(-1), torch.zeros_like(maxs), maxs)
    return mins, maxs


def view_bbox_consistency_loss(
    prim_geom: torch.Tensor,
    prim_mask: torch.Tensor,
    *,
    rel_tol_weight: float = 1.0,
) -> torch.Tensor:
    """
    Soft 3-view projection consistency on TechDraw AABBs:

      front.width ≈ top.width ; front.height ≈ side.height

    Assumes view order TL/BL/TR = front/top/side (same as split_into_views).
    """
    if prim_geom.ndim != 4 or prim_geom.shape[1] < 3:
        return prim_geom.new_zeros(())
    mins, maxs = _view_aabb(prim_geom, prim_mask)
    sizes = (maxs - mins).clamp(min=0.0)  # (B,V,2)
    front_w, front_h = sizes[:, 0, 0], sizes[:, 0, 1]
    top_w, top_h = sizes[:, 1, 0], sizes[:, 1, 1]
    side_w, side_h = sizes[:, 2, 0], sizes[:, 2, 1]
    valid = prim_mask[:, :3].any(dim=-1).all(dim=-1).float()  # (B,)
    if valid.sum() < 1:
        return prim_geom.new_zeros(())
    loss_w = (front_w - top_w).abs()
    loss_h = (front_h - side_h).abs()
    # normalize by mean size to keep scale-ish invariant
    scale = (front_w + front_h + top_w + side_h).clamp(min=1e-3) * 0.25
    loss = ((loss_w + loss_h) / scale) * valid
    return rel_tol_weight * loss.mean()


def surf_type_ce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    ignore_index: int = -1,
) -> torch.Tensor:
    """
    Optional surface-type token CE.

    ``logits``: (B, T, C) or (N, C); ``targets``: long ids.
    When GT surf types are unavailable, callers should skip this term.
    """
    if targets is None or targets.numel() == 0:
        return logits.new_zeros(())
    if logits.ndim == 3:
        c = logits.shape[-1]
        return F.cross_entropy(
            logits.reshape(-1, c),
            targets.reshape(-1),
            ignore_index=ignore_index,
        )
    return F.cross_entropy(logits, targets, ignore_index=ignore_index)


def compute_aux_losses(
    *,
    prim_geom: Optional[torch.Tensor] = None,
    prim_mask: Optional[torch.Tensor] = None,
    enable_view_bbox: bool = False,
    view_bbox_weight: float = 0.1,
    surf_logits: Optional[torch.Tensor] = None,
    surf_targets: Optional[torch.Tensor] = None,
    enable_surf_type: bool = False,
    surf_type_weight: float = 0.1,
) -> AuxLossResult:
    device = None
    for t in (prim_geom, surf_logits):
        if t is not None:
            device = t.device
            break
    zero = torch.zeros((), device=device) if device is not None else torch.zeros(())
    vb = zero
    st = zero
    if enable_view_bbox and prim_geom is not None and prim_mask is not None:
        vb = view_bbox_consistency_loss(prim_geom, prim_mask) * float(view_bbox_weight)
    if enable_surf_type and surf_logits is not None and surf_targets is not None:
        st = surf_type_ce_loss(surf_logits, surf_targets) * float(surf_type_weight)
    total = vb + st
    return AuxLossResult(
        total=total,
        view_bbox=float(vb.detach().item()) if torch.is_tensor(vb) else 0.0,
        surf_type=float(st.detach().item()) if torch.is_tensor(st) else 0.0,
    )


__all__ = [
    "AuxLossResult",
    "view_bbox_consistency_loss",
    "surf_type_ce_loss",
    "compute_aux_losses",
]
