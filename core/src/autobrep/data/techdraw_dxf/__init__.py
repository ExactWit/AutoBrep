"""TechDraw geometric primitives from DXF + SVG (3-view split)."""

from autobrep.data.techdraw_dxf.extract import extract_dxf_primitives
from autobrep.data.techdraw_dxf.extract_svg import extract_svg_primitives
from autobrep.data.techdraw_dxf.filter_merge import filter_and_merge
from autobrep.data.techdraw_dxf.loops import assign_loop_groups
from autobrep.data.techdraw_dxf.schema import (
    GEOM_DIM,
    MAX_PRIMS,
    MAX_PRIMS_PER_VIEW,
    NUM_LINETYPES,
    NUM_PRIM_TYPES,
    NUM_TD_VIEWS,
)
from autobrep.data.techdraw_dxf.split_views import merge_dxfir, split_into_views
from autobrep.data.techdraw_dxf.tensorize import (
    empty_dxf_tensors,
    empty_dxf_view_tensors,
    tensorize_dxf,
    tensorize_dxf_views,
    tensors_to_torch,
)

__all__ = [
    "extract_dxf_primitives",
    "extract_svg_primitives",
    "filter_and_merge",
    "assign_loop_groups",
    "merge_dxfir",
    "split_into_views",
    "tensorize_dxf",
    "tensorize_dxf_views",
    "empty_dxf_tensors",
    "empty_dxf_view_tensors",
    "tensors_to_torch",
    "GEOM_DIM",
    "MAX_PRIMS",
    "MAX_PRIMS_PER_VIEW",
    "NUM_PRIM_TYPES",
    "NUM_LINETYPES",
    "NUM_TD_VIEWS",
]
