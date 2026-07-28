"""Filter short/junk primitives and merge near-colinear line segments."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from autobrep.data.techdraw_dxf.extract import _collect_points
from autobrep.data.techdraw_dxf.schema import DxfIR, PrimIR


def _prim_length(prim: PrimIR) -> float:
    if prim.type == "line":
        a = np.asarray(prim.params["start"][:2], dtype=np.float64)
        b = np.asarray(prim.params["end"][:2], dtype=np.float64)
        return float(np.linalg.norm(b - a))
    if prim.type == "circle":
        return 2.0 * math.pi * float(prim.params.get("radius", 0.0))
    if prim.type == "arc":
        r = float(prim.params.get("radius", 0.0))
        a0 = float(prim.params.get("start_angle", 0.0))
        a1 = float(prim.params.get("end_angle", 0.0))
        return abs(a1 - a0) * r
    pts = _collect_points(prim)
    if len(pts) < 2:
        return 0.0
    arr = np.asarray(pts, dtype=np.float64)
    return float(np.linalg.norm(arr[1:] - arr[:-1], axis=1).sum())


def filter_prims(
    prims: Sequence[PrimIR],
    *,
    min_length: float = 1e-4,
    drop_layers: frozenset[str] | None = None,
) -> list[PrimIR]:
    """
    Drop near-zero fragments and optional layer junk.

    Hidden linetypes are **kept** (needed for TechDraw semantics).
    """
    drop_layers = drop_layers or frozenset({"DEFPOINTS", "ANNOTATION", "DIM", "DIMENSION"})
    out: list[PrimIR] = []
    for p in prims:
        layer = str((p.params or {}).get("layer", "")).upper()
        if layer and any(tok in layer for tok in drop_layers):
            continue
        if _prim_length(p) < min_length and p.type not in {"circle"}:
            continue
        out.append(p)
    return out


def _line_endpoints(prim: PrimIR) -> tuple[np.ndarray, np.ndarray] | None:
    if prim.type != "line":
        return None
    a = np.asarray(prim.params["start"][:2], dtype=np.float64)
    b = np.asarray(prim.params["end"][:2], dtype=np.float64)
    return a, b


def merge_colinear(
    prims: Sequence[PrimIR],
    *,
    angle_tol_deg: float = 3.0,
    gap_tol: float | None = None,
    dist_tol: float | None = None,
) -> list[PrimIR]:
    """
    Greedy merge of nearly colinear, nearly touching LINE segments.

    Non-line primitives are passed through unchanged. Hidden/center styles preserved
    only when both sides share the same linetype.
    """
    lines: list[PrimIR] = []
    others: list[PrimIR] = []
    for p in prims:
        if p.type == "line":
            lines.append(p)
        else:
            others.append(p)
    if len(lines) < 2:
        return list(prims)

    # Estimate sheet scale from extents for relative tolerances.
    pts = []
    for p in prims:
        pts.extend(_collect_points(p))
    if pts:
        arr = np.asarray(pts, dtype=np.float64)
        span = float(np.max(arr.max(axis=0) - arr.min(axis=0)))
    else:
        span = 1.0
    gap_tol = float(gap_tol if gap_tol is not None else max(1e-3, 1e-3 * span))
    dist_tol = float(dist_tol if dist_tol is not None else gap_tol)
    cos_tol = math.cos(math.radians(angle_tol_deg))

    used = [False] * len(lines)
    merged: list[PrimIR] = []

    for i, a in enumerate(lines):
        if used[i]:
            continue
        ep = _line_endpoints(a)
        if ep is None:
            used[i] = True
            merged.append(a)
            continue
        p0, p1 = ep
        direction = p1 - p0
        nrm = float(np.linalg.norm(direction))
        if nrm < 1e-12:
            used[i] = True
            continue
        direction = direction / nrm
        changed = True
        while changed:
            changed = False
            for j, b in enumerate(lines):
                if used[j] or j == i:
                    continue
                if a.linetype != b.linetype:
                    continue
                bep = _line_endpoints(b)
                if bep is None:
                    continue
                q0, q1 = bep
                bd = q1 - q0
                bn = float(np.linalg.norm(bd))
                if bn < 1e-12:
                    used[j] = True
                    continue
                bd = bd / bn
                if abs(float(np.dot(direction, bd))) < cos_tol:
                    continue
                # Point-line distance for both endpoints of b
                def _dist_to_seg_line(pt: np.ndarray) -> float:
                    # distance to infinite line through p0 along direction
                    return float(np.linalg.norm((pt - p0) - np.dot(pt - p0, direction) * direction))

                if max(_dist_to_seg_line(q0), _dist_to_seg_line(q1)) > dist_tol:
                    continue
                # Gap: nearest endpoint pair
                gaps = [
                    float(np.linalg.norm(p0 - q0)),
                    float(np.linalg.norm(p0 - q1)),
                    float(np.linalg.norm(p1 - q0)),
                    float(np.linalg.norm(p1 - q1)),
                ]
                # Also allow overlap (projection overlap)
                projs = [
                    float(np.dot(p0 - p0, direction)),
                    float(np.dot(p1 - p0, direction)),
                    float(np.dot(q0 - p0, direction)),
                    float(np.dot(q1 - p0, direction)),
                ]
                amin, amax = min(projs[0], projs[1]), max(projs[0], projs[1])
                bmin, bmax = min(projs[2], projs[3]), max(projs[2], projs[3])
                overlap = min(amax, bmax) - max(amin, bmin)
                if min(gaps) > gap_tol and overlap < -gap_tol:
                    continue
                # Merge by projecting all 4 endpoints onto direction
                all_pts = [p0, p1, q0, q1]
                ts = [float(np.dot(pt - p0, direction)) for pt in all_pts]
                t0, t1 = min(ts), max(ts)
                new_start = (p0 + t0 * direction).tolist()
                new_end = (p0 + t1 * direction).tolist()
                a = PrimIR(
                    type="line",
                    linetype=a.linetype,
                    params={"start": new_start, "end": new_end, **{k: v for k, v in a.params.items() if k not in {"start", "end"}}},
                    group_id=a.group_id,
                    group_role=a.group_role,
                )
                p0 = np.asarray(new_start, dtype=np.float64)
                p1 = np.asarray(new_end, dtype=np.float64)
                direction = p1 - p0
                nrm = float(np.linalg.norm(direction))
                if nrm < 1e-12:
                    break
                direction = direction / nrm
                used[j] = True
                changed = True
        used[i] = True
        merged.append(a)

    return others + merged


def filter_and_merge(dxfir: DxfIR, **kwargs) -> DxfIR:
    prims = filter_prims(list(dxfir.prims[: int(dxfir.n_prims)]), **kwargs)
    prims = merge_colinear(prims)
    if not prims:
        return DxfIR(
            n_prims=0,
            prims=[],
            bbox_min=np.zeros(2, dtype=np.float32),
            bbox_max=np.ones(2, dtype=np.float32),
        )
    pts = []
    for p in prims:
        pts.extend(_collect_points(p))
    arr = np.asarray(pts, dtype=np.float32) if pts else np.zeros((1, 2), np.float32)
    return DxfIR(
        n_prims=len(prims),
        prims=prims,
        bbox_min=arr.min(axis=0).astype(np.float32),
        bbox_max=arr.max(axis=0).astype(np.float32),
    )


__all__ = ["filter_prims", "merge_colinear", "filter_and_merge"]
