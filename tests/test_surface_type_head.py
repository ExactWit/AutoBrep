"""Minimal tests for surface-type head and id map."""

from __future__ import annotations

import torch

from autobrep.data.surf_types import surf_type_name_to_id, SURF_TYPE_TO_ID
from autobrep.models.surface_type_head import SurfaceTypeHead


def test_surf_type_ids():
    assert surf_type_name_to_id("plane") == 0
    assert surf_type_name_to_id("Cylinder") == 1
    assert "bspline" in SURF_TYPE_TO_ID


def test_surf_type_head_shape():
    head = SurfaceTypeHead(in_dim=32, max_faces=8, n_classes=5)
    x = torch.randn(2, 16, 32)
    logits = head(x)
    assert logits.shape == (2, 8, 5)
