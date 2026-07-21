"""DXF primitive IR schema for structured tech-draw encoding."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

PRIM_LINE = 0
PRIM_ARC = 1
PRIM_CIRCLE = 2
PRIM_SPLINE = 3
PRIM_ELLIPSE = 4
PRIM_LWPOLYLINE = 5
PRIM_OTHER = 6
NUM_PRIM_TYPES = 7

PRIM_TYPE_NAMES = {
    PRIM_LINE: "line",
    PRIM_ARC: "arc",
    PRIM_CIRCLE: "circle",
    PRIM_SPLINE: "spline",
    PRIM_ELLIPSE: "ellipse",
    PRIM_LWPOLYLINE: "lwpolyline",
    PRIM_OTHER: "other",
}
NAME_TO_PRIM_TYPE = {name: idx for idx, name in PRIM_TYPE_NAMES.items()}

LINETYPE_SOLID = 0
LINETYPE_HIDDEN = 1
LINETYPE_CENTER = 2
LINETYPE_OTHER = 3
NUM_LINETYPES = 4

LINETYPE_NAMES = {
    LINETYPE_SOLID: "solid",
    LINETYPE_HIDDEN: "hidden",
    LINETYPE_CENTER: "center",
    LINETYPE_OTHER: "other",
}

# Unified geometry slot layout (12 floats):
# line:        x0,y0,x1,y1, 0...
# arc/circle:  cx,cy,r, start_angle, end_angle (circle: 0, 2pi), 0...
# ellipse:     cx,cy, major_x,major_y, ratio, start_param, end_param, 0...
# spline:      up to 4 control points xy (8) + pad
# lwpolyline:  start/end + midpoint xy (6) + closed flag
GEOM_DIM = 12
MAX_PRIMS = 256
SPLINE_SAMPLE_POINTS = 4


@dataclass
class PrimIR:
    type: str
    linetype: str = "solid"
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def type_id(self) -> int:
        return NAME_TO_PRIM_TYPE.get(self.type, PRIM_OTHER)

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "linetype": self.linetype, "params": self.params}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PrimIR":
        return cls(
            type=str(payload.get("type", "other")),
            linetype=str(payload.get("linetype", "solid")),
            params=dict(payload.get("params") or {}),
        )


@dataclass
class DxfIR:
    n_prims: int
    prims: list[PrimIR]
    bbox_min: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    bbox_max: np.ndarray = field(default_factory=lambda: np.ones(2, dtype=np.float32))

    def to_dict(self) -> dict[str, Any]:
        n = int(self.n_prims)
        return {
            "n_prims": n,
            "prims": [p.to_dict() for p in self.prims[:n]],
            "bbox_min": np.asarray(self.bbox_min, dtype=np.float32).tolist(),
            "bbox_max": np.asarray(self.bbox_max, dtype=np.float32).tolist(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DxfIR":
        prims = [PrimIR.from_dict(item) for item in payload.get("prims") or []]
        n_prims = int(payload.get("n_prims", len(prims)))
        return cls(
            n_prims=n_prims,
            prims=prims,
            bbox_min=np.asarray(payload.get("bbox_min") or [0, 0], dtype=np.float32),
            bbox_max=np.asarray(payload.get("bbox_max") or [1, 1], dtype=np.float32),
        )


__all__ = [
    "PrimIR",
    "DxfIR",
    "PRIM_LINE",
    "PRIM_ARC",
    "PRIM_CIRCLE",
    "PRIM_SPLINE",
    "PRIM_ELLIPSE",
    "PRIM_LWPOLYLINE",
    "PRIM_OTHER",
    "NUM_PRIM_TYPES",
    "PRIM_TYPE_NAMES",
    "NAME_TO_PRIM_TYPE",
    "LINETYPE_SOLID",
    "LINETYPE_HIDDEN",
    "LINETYPE_CENTER",
    "LINETYPE_OTHER",
    "NUM_LINETYPES",
    "LINETYPE_NAMES",
    "GEOM_DIM",
    "MAX_PRIMS",
    "SPLINE_SAMPLE_POINTS",
    "asdict",
]
