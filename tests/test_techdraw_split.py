"""Tests for TechDraw hist-split, filter/merge, and loop grouping."""

from __future__ import annotations

import numpy as np

from autobrep.data.techdraw_dxf.filter_merge import filter_prims, merge_colinear
from autobrep.data.techdraw_dxf.loops import assign_loop_groups
from autobrep.data.techdraw_dxf.schema import (
    GROUP_ROLE_OUTER,
    DxfIR,
    PrimIR,
)
from autobrep.data.techdraw_dxf.split_views import split_into_views
from autobrep.data.techdraw_dxf.tensorize import tensorize_dxf


def _box(cx: float, cy: float, w: float, h: float, *, lt: str = "solid") -> list[PrimIR]:
    x0, x1 = cx - w / 2, cx + w / 2
    y0, y1 = cy - h / 2, cy + h / 2
    return [
        PrimIR(type="line", linetype=lt, params={"start": [x0, y0], "end": [x1, y0]}),
        PrimIR(type="line", linetype=lt, params={"start": [x1, y0], "end": [x1, y1]}),
        PrimIR(type="line", linetype=lt, params={"start": [x1, y1], "end": [x0, y1]}),
        PrimIR(type="line", linetype=lt, params={"start": [x0, y1], "end": [x0, y0]}),
    ]


def _synthetic_sheet() -> DxfIR:
    """
    Classic 3-view sheet:
      TL front @ (0, 20) size 10x8
      BL top   @ (0, 0)  size 10x6
      TR side  @ (20, 20) size 6x8
    """
    prims = []
    prims.extend(_box(0.0, 20.0, 10.0, 8.0))
    prims.extend(_box(0.0, 0.0, 10.0, 6.0))
    prims.extend(_box(20.0, 20.0, 6.0, 8.0))
    # hidden line kept
    prims.append(
        PrimIR(
            type="line",
            linetype="hidden",
            params={"start": [-2.0, 20.0], "end": [2.0, 20.0]},
        )
    )
    pts = []
    for p in prims:
        pts.extend([p.params["start"], p.params["end"]])
    arr = np.asarray(pts, dtype=np.float32)
    return DxfIR(
        n_prims=len(prims),
        prims=prims,
        bbox_min=arr.min(0),
        bbox_max=arr.max(0),
    )


def test_hist_split_three_nonempty():
    sheet = _synthetic_sheet()
    views = split_into_views(sheet)
    assert len(views) == 3
    nonempty = sum(1 for v in views if int(v.n_prims) > 0)
    assert nonempty == 3
    # naming stability: TL should be densest near high-y / low-x
    c0 = np.mean(
        [np.mean([p.params["start"], p.params["end"]], 0) for p in views[0].prims if p.type == "line"],
        axis=0,
    )
    assert float(c0[1]) > 10.0


def test_naming_stable_across_shuffle():
    sheet = _synthetic_sheet()
    a = split_into_views(sheet)
    # shuffle prim order
    rng = np.random.default_rng(0)
    order = rng.permutation(int(sheet.n_prims))
    shuffled = DxfIR(
        n_prims=sheet.n_prims,
        prims=[sheet.prims[i] for i in order],
        bbox_min=sheet.bbox_min,
        bbox_max=sheet.bbox_max,
    )
    b = split_into_views(shuffled)
    for va, vb in zip(a, b):
        assert abs(int(va.n_prims) - int(vb.n_prims)) <= 2


def test_filter_keeps_hidden_drops_short():
    prims = [
        PrimIR(type="line", linetype="hidden", params={"start": [0, 0], "end": [1, 0]}),
        PrimIR(type="line", linetype="solid", params={"start": [0, 0], "end": [1e-9, 0]}),
    ]
    out = filter_prims(prims, min_length=1e-6)
    assert len(out) == 1
    assert out[0].linetype == "hidden"


def test_merge_colinear():
    prims = [
        PrimIR(type="line", linetype="solid", params={"start": [0, 0], "end": [1, 0]}),
        PrimIR(type="line", linetype="solid", params={"start": [1.001, 0], "end": [2, 0]}),
    ]
    out = merge_colinear(prims, gap_tol=0.01, dist_tol=0.01)
    lines = [p for p in out if p.type == "line"]
    assert len(lines) == 1


def test_loop_groups_square():
    prims = _box(0, 0, 2, 2)
    grouped = assign_loop_groups(prims)
    assert len(grouped) == 4
    assert all(g.group_id == grouped[0].group_id for g in grouped)
    assert grouped[0].group_role in {GROUP_ROLE_OUTER, "outer"}


def test_tensorize_norm_range_and_groups():
    prims = assign_loop_groups(_box(5, 5, 2, 2))
    dxfir = DxfIR(
        n_prims=len(prims),
        prims=prims,
        bbox_min=np.array([4, 4], dtype=np.float32),
        bbox_max=np.array([6, 6], dtype=np.float32),
    )
    t = tensorize_dxf(dxfir, max_prims=16)
    geom = t["prim_geom"][t["prim_mask"]]
    assert geom[:, 0:4].min() >= -1.01
    assert geom[:, 0:4].max() <= 1.01
    assert "prim_group_ids" in t
    assert t["prim_group_ids"][t["prim_mask"]].sum() >= 0
