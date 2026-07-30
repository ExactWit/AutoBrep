"""Surface-type id map shared by data cache and SurfaceTypeHead."""

from __future__ import annotations

SURF_TYPE_NAMES = ("plane", "cylinder", "cone", "sphere", "bspline")
SURF_TYPE_TO_ID = {n: i for i, n in enumerate(SURF_TYPE_NAMES)}
NUM_SURF_TYPES = len(SURF_TYPE_NAMES)
SURF_TYPE_PAD = -1
SURF_TYPE_MAX_FACES = 64


def surf_type_name_to_id(name: str) -> int:
    key = str(name).strip().lower()
    if key in SURF_TYPE_TO_ID:
        return SURF_TYPE_TO_ID[key]
    if key in {"bspline_surface", "b_spline", "nurbs"}:
        return SURF_TYPE_TO_ID["bspline"]
    return SURF_TYPE_TO_ID["bspline"]
