"""TechDraw DXF structured primitives (ported from eccv26)."""

from autobrep.data.techdraw_dxf.extract import extract_dxf_primitives
from autobrep.data.techdraw_dxf.schema import GEOM_DIM, MAX_PRIMS, NUM_LINETYPES, NUM_PRIM_TYPES
from autobrep.data.techdraw_dxf.tensorize import (
    empty_dxf_tensors,
    tensorize_dxf,
    tensors_to_torch,
)

__all__ = [
    "extract_dxf_primitives",
    "tensorize_dxf",
    "empty_dxf_tensors",
    "tensors_to_torch",
    "GEOM_DIM",
    "MAX_PRIMS",
    "NUM_PRIM_TYPES",
    "NUM_LINETYPES",
]
