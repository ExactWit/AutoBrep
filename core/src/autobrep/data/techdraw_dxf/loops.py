"""Contour / loop grouping over TechDraw primitives (endpoint adjacency)."""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import numpy as np

from autobrep.data.techdraw_dxf.schema import (
    GROUP_ROLE_INNER,
    GROUP_ROLE_ISOLATED,
    GROUP_ROLE_OUTER,
    PrimIR,
)


def _endpoints(prim: PrimIR) -> list[np.ndarray]:
    params = prim.params
    if prim.type == "line":
        return [
            np.asarray(params["start"][:2], dtype=np.float64),
            np.asarray(params["end"][:2], dtype=np.float64),
        ]
    if prim.type == "arc":
        c = np.asarray(params["center"][:2], dtype=np.float64)
        r = float(params["radius"])
        a0 = float(params.get("start_angle", 0.0))
        a1 = float(params.get("end_angle", 2.0 * np.pi))
        return [
            c + r * np.array([np.cos(a0), np.sin(a0)]),
            c + r * np.array([np.cos(a1), np.sin(a1)]),
        ]
    if prim.type == "lwpolyline":
        pts = list(params.get("points") or [])
        if not pts:
            return []
        if params.get("closed") and len(pts) >= 2:
            # closed: treat as self-loop endpoints coincide at first point
            p0 = np.asarray(pts[0][:2], dtype=np.float64)
            return [p0, p0]
        return [
            np.asarray(pts[0][:2], dtype=np.float64),
            np.asarray(pts[-1][:2], dtype=np.float64),
        ]
    if prim.type == "circle":
        c = np.asarray(params["center"][:2], dtype=np.float64)
        return [c, c]  # closed self
    return []


def _quantize(pt: np.ndarray, tol: float) -> tuple[int, int]:
    return (int(round(float(pt[0]) / tol)), int(round(float(pt[1]) / tol)))


def assign_loop_groups(
    prims: Sequence[PrimIR],
    *,
    snap_tol: float | None = None,
) -> list[PrimIR]:
    """
    Assign ``group_id`` / ``group_role`` via endpoint adjacency + cycle search.

    - Circles / closed polylines → own group (outer if largest bbox area else inner).
    - Connected line/arc chains that close → loop groups.
    - Remainder → isolated.
    """
    if not prims:
        return []

    pts_all = []
    for p in prims:
        pts_all.extend(_endpoints(p))
    if pts_all:
        arr = np.stack(pts_all, axis=0)
        span = float(np.max(arr.max(axis=0) - arr.min(axis=0)))
    else:
        span = 1.0
    tol = float(snap_tol if snap_tol is not None else max(1e-4, 1e-3 * span))

    n = len(prims)
    # Build undirected graph: node = prim index for open segments; closed get own component
    adj: dict[int, list[int]] = defaultdict(list)
    node_of: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    # (prim_idx, end_idx 0|1)
    closed_idxs: list[int] = []

    for i, p in enumerate(prims):
        eps = _endpoints(p)
        if len(eps) < 2:
            continue
        if np.linalg.norm(eps[0] - eps[1]) < tol:
            closed_idxs.append(i)
            continue
        for e_i, pt in enumerate(eps[:2]):
            node_of[_quantize(pt, tol)].append((i, e_i))

    for _key, items in node_of.items():
        # connect all prims sharing a snapped endpoint
        for a in range(len(items)):
            for b in range(a + 1, len(items)):
                i, _ = items[a]
                j, _ = items[b]
                if i == j:
                    continue
                adj[i].append(j)
                adj[j].append(i)

    visited = [False] * n
    components: list[list[int]] = []

    for i in closed_idxs:
        if not visited[i]:
            visited[i] = True
            components.append([i])

    for i in range(n):
        if visited[i]:
            continue
        if i not in adj and i not in closed_idxs:
            visited[i] = True
            components.append([i])
            continue
        stack = [i]
        visited[i] = True
        comp = []
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj.get(u, []):
                if not visited[v]:
                    visited[v] = True
                    stack.append(v)
        components.append(comp)

    def _comp_bbox_area(idxs: list[int]) -> float:
        pts = []
        for i in idxs:
            pts.extend(_endpoints(prims[i]))
            # also use circle radius
            p = prims[i]
            if p.type == "circle":
                c = np.asarray(p.params["center"][:2], dtype=np.float64)
                r = float(p.params["radius"])
                pts.extend([c - r, c + r])
        if not pts:
            return 0.0
        arr = np.stack(pts, axis=0)
        wh = arr.max(axis=0) - arr.min(axis=0)
        return float(wh[0] * wh[1])

    def _is_closed_comp(idxs: list[int]) -> bool:
        if len(idxs) == 1 and idxs[0] in closed_idxs:
            return True
        # degree-2 cycle heuristic: every open endpoint degree matches
        if len(idxs) < 2:
            return False
        # count endpoint degrees
        deg: dict[tuple[int, int], int] = defaultdict(int)
        for i in idxs:
            eps = _endpoints(prims[i])
            if len(eps) < 2:
                return False
            if np.linalg.norm(eps[0] - eps[1]) < tol:
                continue
            for pt in eps[:2]:
                deg[_quantize(pt, tol)] += 1
        return bool(deg) and all(v >= 2 for v in deg.values())

    areas = [_comp_bbox_area(c) for c in components]
    max_area = max(areas) if areas else 0.0

    out: list[PrimIR] = [None] * n  # type: ignore
    for gid, (comp, area) in enumerate(zip(components, areas)):
        closed = _is_closed_comp(comp)
        if not closed:
            role = GROUP_ROLE_ISOLATED
        elif area >= max_area * 0.98 and max_area > 0:
            role = GROUP_ROLE_OUTER
        else:
            role = GROUP_ROLE_INNER
        for i in comp:
            p = prims[i]
            out[i] = PrimIR(
                type=p.type,
                linetype=p.linetype,
                params=dict(p.params),
                group_id=int(gid),
                group_role=str(role),
            )
    return [p for p in out if p is not None]


__all__ = ["assign_loop_groups"]
