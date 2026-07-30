"""Surface-type classification head (Stage A aux; Stage C extends to params)."""

from __future__ import annotations

import torch
import torch.nn as nn

from autobrep.data.surf_types import (
    NUM_SURF_TYPES,
    SURF_TYPE_MAX_FACES,
    SURF_TYPE_NAMES,
    SURF_TYPE_PAD,
    SURF_TYPE_TO_ID,
    surf_type_name_to_id,
)


class SurfaceTypeHead(nn.Module):
    """Predict per-face surface type from pooled condition memory."""

    def __init__(
        self,
        in_dim: int,
        max_faces: int = SURF_TYPE_MAX_FACES,
        n_classes: int = NUM_SURF_TYPES,
    ):
        super().__init__()
        self.max_faces = int(max_faces)
        self.n_classes = int(n_classes)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, max_faces * n_classes),
        )

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            cond: (B, M, D) or (B, D) condition tokens / pooled vector
        Returns:
            logits: (B, max_faces, n_classes)
        """
        if cond.ndim == 3:
            pooled = cond.mean(dim=1)
        else:
            pooled = cond
        b = pooled.shape[0]
        return self.mlp(pooled).view(b, self.max_faces, self.n_classes)


__all__ = [
    "SURF_TYPE_NAMES",
    "SURF_TYPE_TO_ID",
    "NUM_SURF_TYPES",
    "SURF_TYPE_PAD",
    "SURF_TYPE_MAX_FACES",
    "surf_type_name_to_id",
    "SurfaceTypeHead",
]
