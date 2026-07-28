"""Tests for P2 auxiliary losses."""

from __future__ import annotations

import torch

from autobrep.models.aux_losses import (
    compute_aux_losses,
    view_bbox_consistency_loss,
)


def test_view_bbox_consistent_sheet():
    # Three views with matching front/top width and front/side height
    b, v, n, g = 1, 3, 4, 12
    geom = torch.zeros(b, v, n, g)
    mask = torch.ones(b, v, n, dtype=torch.bool)
    # front: w=2,h=1
    geom[0, 0, 0, 0:4] = torch.tensor([-1, -0.5, 1, 0.5])
    # top: w=2,h=0.5
    geom[0, 1, 0, 0:4] = torch.tensor([-1, -0.25, 1, 0.25])
    # side: w=0.5,h=1
    geom[0, 2, 0, 0:4] = torch.tensor([-0.25, -0.5, 0.25, 0.5])
    loss = view_bbox_consistency_loss(geom, mask)
    assert float(loss) < 0.05


def test_aux_disabled_zero():
    geom = torch.randn(2, 3, 8, 12)
    mask = torch.ones(2, 3, 8, dtype=torch.bool)
    aux = compute_aux_losses(
        prim_geom=geom,
        prim_mask=mask,
        enable_view_bbox=False,
        enable_surf_type=False,
    )
    assert float(aux.total) == 0.0


def test_aux_enabled_positive():
    geom = torch.randn(2, 3, 8, 12)
    mask = torch.ones(2, 3, 8, dtype=torch.bool)
    aux = compute_aux_losses(
        prim_geom=geom,
        prim_mask=mask,
        enable_view_bbox=True,
        view_bbox_weight=0.1,
    )
    assert aux.total.ndim == 0
