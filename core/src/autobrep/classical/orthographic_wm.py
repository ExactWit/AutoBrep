"""Wesley–Markowsky style bottom-up reconstruction from 3 orthographic views.

Pipeline (practical subset of the classical 5 maps):

  fVR: 2D vertices → 3D candidate vertices (coordinate matching)
  fED: 3D vertices → 3D edges (back-projection filter)
  fFA: 3D edges → planar candidate faces (left-turn minimal loops)
  fBL/fSL: sew faces → solid, back-project vs input, keep best

Circles in views are handled as cylinder candidates (Liu-style extension lite).
Ghost elements are rejected at fED and at final back-projection.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np

from autobrep.data.techdraw_dxf.extract import extract_dxf_primitives
from autobrep.data.techdraw_dxf.extract_svg import extract_svg_primitives
from autobrep.data.techdraw_dxf.filter_merge import filter_and_merge
from autobrep.data.techdraw_dxf.schema import DxfIR, PrimIR
from autobrep.data.techdraw_dxf.split_views import merge_dxfir, split_into_views


# ---------------------------------------------------------------------------
# 2D primitives
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Seg2D:
    a: tuple[float, float]
    b: tuple[float, float]
    linetype: str = "solid"

    def length(self) -> float:
        return float(math.hypot(self.b[0] - self.a[0], self.b[1] - self.a[1]))


@dataclass(frozen=True)
class Circle2D:
    c: tuple[float, float]
    r: float
    linetype: str = "solid"


@dataclass
class View2D:
    name: str  # front | top | side
    segs: list[Seg2D] = field(default_factory=list)
    circles: list[Circle2D] = field(default_factory=list)
    bbox_min: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float64))
    bbox_max: np.ndarray = field(default_factory=lambda: np.ones(2, dtype=np.float64))


def _q(pt: tuple[float, float] | np.ndarray, tol: float) -> tuple[int, int]:
    return (int(round(float(pt[0]) / tol)), int(round(float(pt[1]) / tol)))


def _prim_to_segs_circles(prim: PrimIR) -> tuple[list[Seg2D], list[Circle2D]]:
    segs: list[Seg2D] = []
    circles: list[Circle2D] = []
    p = prim.params
    lt = prim.linetype or "solid"
    if prim.type == "line":
        a = (float(p["start"][0]), float(p["start"][1]))
        b = (float(p["end"][0]), float(p["end"][1]))
        if math.hypot(b[0] - a[0], b[1] - a[1]) > 1e-9:
            segs.append(Seg2D(a, b, lt))
    elif prim.type == "lwpolyline":
        pts = [(float(x[0]), float(x[1])) for x in (p.get("points") or [])]
        if len(pts) >= 2:
            n = len(pts)
            lim = n if p.get("closed") else n - 1
            for i in range(lim):
                a, b = pts[i], pts[(i + 1) % n]
                if math.hypot(b[0] - a[0], b[1] - a[1]) > 1e-9:
                    segs.append(Seg2D(a, b, lt))
    elif prim.type == "circle":
        c = (float(p["center"][0]), float(p["center"][1]))
        circles.append(Circle2D(c, float(p["radius"]), lt))
    elif prim.type == "arc":
        # Chord approximation for wireframe pairing; cylinder path uses circles.
        c = np.asarray(p["center"][:2], dtype=np.float64)
        r = float(p["radius"])
        a0 = float(p.get("start_angle", 0.0))
        a1 = float(p.get("end_angle", 2.0 * math.pi))
        if a1 < a0:
            a1 += 2.0 * math.pi
        n = max(4, int(math.ceil(abs(a1 - a0) / (math.pi / 6))))
        pts = [
            (float(c[0] + r * math.cos(a0 + (a1 - a0) * i / n)),
             float(c[1] + r * math.sin(a0 + (a1 - a0) * i / n)))
            for i in range(n + 1)
        ]
        for i in range(len(pts) - 1):
            segs.append(Seg2D(pts[i], pts[i + 1], lt))
    return segs, circles


def _view_from_dxfir(name: str, dxfir: DxfIR) -> View2D:
    segs: list[Seg2D] = []
    circles: list[Circle2D] = []
    for prim in dxfir.prims[: int(dxfir.n_prims)]:
        s, c = _prim_to_segs_circles(prim)
        segs.extend(s)
        circles.extend(c)
    return View2D(
        name=name,
        segs=segs,
        circles=circles,
        bbox_min=np.asarray(dxfir.bbox_min, dtype=np.float64),
        bbox_max=np.asarray(dxfir.bbox_max, dtype=np.float64),
    )


def label_orthographic_views(views: Sequence[DxfIR]) -> dict[str, View2D]:
    """Assign Front / Top / Side from sheet layout centroids.

    Assumes third-angle-like L-layout common in this dataset:
      Top above Front, Side to the right of Front.
    """
    nonempty = [(i, v) for i, v in enumerate(views) if int(v.n_prims) > 0]
    if len(nonempty) < 2:
        raise ValueError(f"need ≥2 nonempty views, got {len(nonempty)}")

    cents = []
    for i, v in nonempty:
        c = 0.5 * (np.asarray(v.bbox_min, dtype=np.float64) + np.asarray(v.bbox_max, dtype=np.float64))
        cents.append((i, c))

    # Side = rightmost
    side_i = max(cents, key=lambda t: float(t[1][0]))[0]
    rest = [t for t in cents if t[0] != side_i]
    if len(rest) == 1:
        front_i = rest[0][0]
        top_i = rest[0][0]
    else:
        # Top = higher sheet-y among remaining; Front = lower
        top_i = max(rest, key=lambda t: float(t[1][1]))[0]
        front_i = min(rest, key=lambda t: float(t[1][1]))[0]

    out = {
        "front": _view_from_dxfir("front", views[front_i]),
        "top": _view_from_dxfir("top", views[top_i]),
        "side": _view_from_dxfir("side", views[side_i]),
    }
    return out


def load_labeled_views(
    dataset_root: Path,
    *,
    dxf_rel: str = "",
    svg_rel: str = "",
) -> dict[str, View2D]:
    root = Path(dataset_root)
    parts: list[DxfIR] = []
    if dxf_rel:
        p = root / dxf_rel
        if p.is_file():
            parts.append(extract_dxf_primitives(p))
    if svg_rel:
        p = root / svg_rel
        if p.is_file():
            try:
                parts.append(extract_svg_primitives(p))
            except Exception:  # noqa: BLE001
                pass
    if not parts:
        raise FileNotFoundError(f"no TechDraw for dxf={dxf_rel!r} svg={svg_rel!r}")
    merged = filter_and_merge(merge_dxfir(parts))
    views = split_into_views(merged)
    return label_orthographic_views(views)


# ---------------------------------------------------------------------------
# Local view coordinates (shared axes aligned via bbox mins)
# ---------------------------------------------------------------------------


def _local_xy(view: View2D, pt: tuple[float, float]) -> tuple[float, float]:
    return (float(pt[0] - view.bbox_min[0]), float(pt[1] - view.bbox_min[1]))


def _view_vertices(view: View2D, tol: float) -> list[tuple[float, float]]:
    buckets: dict[tuple[int, int], list[tuple[float, float]]] = defaultdict(list)
    for seg in view.segs:
        for pt in (seg.a, seg.b):
            lp = _local_xy(view, pt)
            buckets[_q(lp, tol)].append(lp)
    for cir in view.circles:
        # sample 4 cardinal points so fVR can pair cylinder extents
        c = _local_xy(view, cir.c)
        for dx, dy in ((cir.r, 0), (-cir.r, 0), (0, cir.r), (0, -cir.r)):
            lp = (c[0] + dx, c[1] + dy)
            buckets[_q(lp, tol)].append(lp)
    out = []
    for pts in buckets.values():
        arr = np.asarray(pts, dtype=np.float64)
        out.append((float(arr[:, 0].mean()), float(arr[:, 1].mean())))
    return out


def _view_segs_local(view: View2D) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    return [(_local_xy(view, s.a), _local_xy(view, s.b)) for s in view.segs]


# ---------------------------------------------------------------------------
# fVR / fED / fFA
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class V3:
    x: float
    y: float
    z: float

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)

    def as_np(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=np.float64)


def _near(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol


def f_vr(
    front: View2D,
    top: View2D,
    side: View2D,
    *,
    tol: float,
) -> list[V3]:
    """Match Front(x,y) + Top(x,z) + Side(y,z) → 3D candidates.

    Side sheet convention (right of Front): local_u → Z (depth), local_v → Y.
    """
    fv = _view_vertices(front, tol)
    tv = _view_vertices(top, tol)
    # Side: (sheet_u, sheet_v) → (z, y) then store as (y, z) for matching
    sv_raw = _view_vertices(side, tol)
    sv = [(y, z) for z, y in sv_raw]  # (y, z)

    # index by shared coords
    by_x_f: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for x, y in fv:
        by_x_f[int(round(x / tol))].append((x, y))
    by_x_t: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for x, z in tv:
        by_x_t[int(round(x / tol))].append((x, z))

    candidates: list[V3] = []
    seen: set[tuple[int, int, int]] = set()

    # Front+Top → (x,y,z); confirm Side has (y,z)
    for kx, fpts in by_x_f.items():
        for kx2 in (kx - 1, kx, kx + 1):
            for x_f, y in fpts:
                for x_t, z in by_x_t.get(kx2, []):
                    if not _near(x_f, x_t, tol):
                        continue
                    x = 0.5 * (x_f + x_t)
                    if not _side_has(sv, y, z, tol):
                        continue
                    key = (
                        int(round(x / tol)),
                        int(round(y / tol)),
                        int(round(z / tol)),
                    )
                    if key in seen:
                        continue
                    seen.add(key)
                    candidates.append(V3(x, y, z))

    # Front+Side fallback when Top sparse
    by_y_s: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for y, z in sv:
        by_y_s[int(round(y / tol))].append((y, z))
    for x, y in fv:
        ky = int(round(y / tol))
        for ky2 in (ky - 1, ky, ky + 1):
            for y_s, z in by_y_s.get(ky2, []):
                if not _near(y, y_s, tol):
                    continue
                if not _top_has(tv, x, z, tol):
                    continue
                key = (
                    int(round(x / tol)),
                    int(round(y / tol)),
                    int(round(z / tol)),
                )
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(V3(x, y, z))

    return candidates


def _side_local_yz(side: View2D, pt: tuple[float, float]) -> tuple[float, float]:
    """Side sheet (u,v) → (y,z) with u=depth/Z, v=height/Y."""
    u, v = _local_xy(side, pt)
    return (v, u)


def _side_segs_local(side: View2D) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    return [(_side_local_yz(side, s.a), _side_local_yz(side, s.b)) for s in side.segs]


def _side_has(sv: list[tuple[float, float]], y: float, z: float, tol: float) -> bool:
    for yy, zz in sv:
        if _near(yy, y, tol) and _near(zz, z, tol):
            return True
    return False


def _top_has(tv: list[tuple[float, float]], x: float, z: float, tol: float) -> bool:
    for xx, zz in tv:
        if _near(xx, x, tol) and _near(zz, z, tol):
            return True
    return False


def _seg_covers(
    segs: Sequence[tuple[tuple[float, float], tuple[float, float]]],
    p0: tuple[float, float],
    p1: tuple[float, float],
    tol: float,
) -> bool:
    """True if segment p0–p1 lies on some 2D edge (collinear + overlap)."""
    if math.hypot(p1[0] - p0[0], p1[1] - p0[1]) < tol:
        return True
    for a, b in segs:
        if _point_on_seg(p0, a, b, tol) and _point_on_seg(p1, a, b, tol):
            return True
        # also allow p0-p1 to cover a-b (same line support)
        if _point_on_seg(a, p0, p1, tol) and _point_on_seg(b, p0, p1, tol):
            return True
    return False


def _point_on_seg(
    p: tuple[float, float],
    a: tuple[float, float],
    b: tuple[float, float],
    tol: float,
) -> bool:
    ax, ay = a
    bx, by = b
    px, py = p
    abx, aby = bx - ax, by - ay
    apx, apy = px - ax, py - ay
    lab2 = abx * abx + aby * aby
    if lab2 < 1e-18:
        return math.hypot(px - ax, py - ay) <= tol
    t = (apx * abx + apy * aby) / lab2
    if t < -1e-6 or t > 1.0 + 1e-6:
        # allow slight extension for snapped endpoints
        if t < -0.05 or t > 1.05:
            return False
    proj = (ax + t * abx, ay + t * aby)
    return math.hypot(px - proj[0], py - proj[1]) <= tol


def f_ed(
    verts: Sequence[V3],
    front: View2D,
    top: View2D,
    side: View2D,
    *,
    tol: float,
    max_edges: int = 8000,
) -> list[tuple[int, int]]:
    """Pair 3D verts whose projections coincide with 2D edges in all views."""
    fsegs = _view_segs_local(front)
    tsegs = _view_segs_local(top)
    ssegs = _side_segs_local(side)
    n = len(verts)
    edges: list[tuple[int, int]] = []
    # Prefer axis-aligned then short edges to limit explosion
    pairs: list[tuple[float, int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            vi, vj = verts[i], verts[j]
            d = math.sqrt(
                (vi.x - vj.x) ** 2 + (vi.y - vj.y) ** 2 + (vi.z - vj.z) ** 2
            )
            if d < tol:
                continue
            pairs.append((d, i, j))
    pairs.sort(key=lambda t: t[0])
    for d, i, j in pairs:
        if len(edges) >= max_edges:
            break
        vi, vj = verts[i], verts[j]
        # degenerate projection → accept if both ends project to same point
        ok_f = _seg_covers(fsegs, (vi.x, vi.y), (vj.x, vj.y), tol) or (
            _near(vi.x, vj.x, tol) and _near(vi.y, vj.y, tol)
        )
        ok_t = _seg_covers(tsegs, (vi.x, vi.z), (vj.x, vj.z), tol) or (
            _near(vi.x, vj.x, tol) and _near(vi.z, vj.z, tol)
        )
        ok_s = _seg_covers(ssegs, (vi.y, vi.z), (vj.y, vj.z), tol) or (
            _near(vi.y, vj.y, tol) and _near(vi.z, vj.z, tol)
        )
        if ok_f and ok_t and ok_s:
            edges.append((i, j))
    return edges


def f_fa(
    verts: Sequence[V3],
    edges: Sequence[tuple[int, int]],
    *,
    tol: float,
) -> list[list[int]]:
    """Left-turn minimal planar loops on the 3D wireframe."""
    if len(verts) < 3 or not edges:
        return []
    adj: dict[int, list[int]] = defaultdict(list)
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)

    pts = [v.as_np() for v in verts]
    loops: list[list[int]] = []
    seen_cycles: set[frozenset[int]] = set()

    def _plane_normal(i0: int, i1: int, i2: int) -> Optional[np.ndarray]:
        n = np.cross(pts[i1] - pts[i0], pts[i2] - pts[i0])
        ln = float(np.linalg.norm(n))
        if ln < 1e-10:
            return None
        return n / ln

    for start, nbrs in list(adj.items()):
        for nxt in nbrs:
            # walk left-most in a candidate plane
            for third in adj[nxt]:
                if third == start:
                    continue
                normal = _plane_normal(start, nxt, third)
                if normal is None:
                    continue
                loop = _walk_left(adj, pts, start, nxt, normal, tol)
                if loop is None or len(loop) < 3:
                    continue
                key = frozenset(loop)
                if key in seen_cycles:
                    continue
                # planarity check
                if not _loop_planar(pts, loop, tol):
                    continue
                seen_cycles.add(key)
                loops.append(loop)

    # Prefer smaller loops (minimal)
    loops.sort(key=len)
    # Drop loops that are unions of smaller ones (simple containment by vertex set)
    minimal: list[list[int]] = []
    for loop in loops:
        s = set(loop)
        if any(set(m) < s for m in minimal):
            continue
        minimal.append(loop)
    return minimal[:200]


def _walk_left(
    adj: dict[int, list[int]],
    pts: list[np.ndarray],
    start: int,
    nxt: int,
    normal: np.ndarray,
    tol: float,
    max_len: int = 32,
) -> Optional[list[int]]:
    path = [start, nxt]
    prev, cur = start, nxt
    for _ in range(max_len):
        best = None
        best_ang = None
        v_in = pts[cur] - pts[prev]
        v_in = v_in / (np.linalg.norm(v_in) + 1e-12)
        for cand in adj[cur]:
            if cand == prev:
                continue
            v_out = pts[cand] - pts[cur]
            if np.linalg.norm(v_out) < tol:
                continue
            # reject off-plane
            if abs(float(np.dot(normal, pts[cand] - pts[start]))) > 5 * tol:
                continue
            v_out = v_out / (np.linalg.norm(v_out) + 1e-12)
            # signed angle around normal
            cross = np.cross(v_in, v_out)
            sin_a = float(np.dot(normal, cross))
            cos_a = float(np.clip(np.dot(v_in, v_out), -1.0, 1.0))
            ang = math.atan2(sin_a, cos_a)  # (-pi, pi]
            # left-most = largest positive turn (ccw w.r.t. normal)
            score = ang
            if best is None or score > best_ang + 1e-9:
                best = cand
                best_ang = score
        if best is None:
            return None
        if best == start:
            return path
        if best in path:
            return None
        path.append(best)
        prev, cur = cur, best
        v_in = pts[cur] - pts[prev]
        v_in = v_in / (np.linalg.norm(v_in) + 1e-12)
    return None


def _loop_planar(pts: list[np.ndarray], loop: list[int], tol: float) -> bool:
    if len(loop) < 3:
        return False
    p0, p1, p2 = pts[loop[0]], pts[loop[1]], pts[loop[2]]
    n = np.cross(p1 - p0, p2 - p0)
    ln = float(np.linalg.norm(n))
    if ln < 1e-10:
        return False
    n = n / ln
    for i in loop:
        if abs(float(np.dot(n, pts[i] - p0))) > 5 * tol:
            return False
    return True


# ---------------------------------------------------------------------------
# OCC solid build + back-projection
# ---------------------------------------------------------------------------


def _build_solid_from_loops(
    verts: Sequence[V3],
    loops: Sequence[Sequence[int]],
    *,
    sewing_tol: float,
):
    from OCC.Core.BRepBuilderAPI import (
        BRepBuilderAPI_MakeEdge,
        BRepBuilderAPI_MakeFace,
        BRepBuilderAPI_MakePolygon,
        BRepBuilderAPI_MakeSolid,
        BRepBuilderAPI_Sewing,
    )
    from OCC.Core.gp import gp_Pnt
    from OCC.Core.TopoDS import TopoDS_Shell, topods

    sewing = BRepBuilderAPI_Sewing(sewing_tol)
    n_faces = 0
    for loop in loops:
        if len(loop) < 3:
            continue
        poly = BRepBuilderAPI_MakePolygon()
        for idx in loop:
            v = verts[idx]
            poly.Add(gp_Pnt(float(v.x), float(v.y), float(v.z)))
        poly.Close()
        if not poly.IsDone():
            continue
        wire = poly.Wire()
        face_mk = BRepBuilderAPI_MakeFace(wire, True)
        if not face_mk.IsDone():
            continue
        sewing.Add(face_mk.Face())
        n_faces += 1
    if n_faces == 0:
        return None, "no_faces"
    sewing.Perform()
    sewn = sewing.SewedShape()
    try:
        if sewn.ShapeType() == 3:  # shell
            shell = topods.Shell(sewn)
            solid_mk = BRepBuilderAPI_MakeSolid(shell)
            if solid_mk.IsDone():
                return solid_mk.Solid(), "sewn_solid"
        if sewn.ShapeType() == 2:  # solid
            return sewn, "sewn_as_solid"
        # compound: try extract shell
        from OCC.Core.TopExp import TopExp_Explorer
        from OCC.Core.TopAbs import TopAbs_SHELL, TopAbs_SOLID

        exp = TopExp_Explorer(sewn, TopAbs_SOLID)
        if exp.More():
            return topods.Solid(exp.Current()), "compound_solid"
        exp = TopExp_Explorer(sewn, TopAbs_SHELL)
        if exp.More():
            shell = topods.Shell(exp.Current())
            solid_mk = BRepBuilderAPI_MakeSolid(shell)
            if solid_mk.IsDone():
                return solid_mk.Solid(), "compound_shell_solid"
    except Exception as exc:  # noqa: BLE001
        return None, f"sew_fail:{exc}"
    return sewn, "sewn_raw"


def _try_box_fallback(verts: Sequence[V3]):
    if not verts:
        return None
    xs = [v.x for v in verts]
    ys = [v.y for v in verts]
    zs = [v.z for v in verts]
    dx = max(xs) - min(xs)
    dy = max(ys) - min(ys)
    dz = max(zs) - min(zs)
    if min(dx, dy, dz) < 1e-6:
        return None
    from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCC.Core.gp import gp_Pnt

    return BRepPrimAPI_MakeBox(
        gp_Pnt(min(xs), min(ys), min(zs)),
        float(dx),
        float(dy),
        float(dz),
    ).Shape()


def _try_cylinders(
    front: View2D,
    top: View2D,
    side: View2D,
    *,
    tol: float,
):
    """Lite cylinder reconstruction: matching circles across views."""
    from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeCylinder
    from OCC.Core.gp import gp_Ax2, gp_Dir, gp_Pnt
    from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Fuse

    cands = []
    # Front circle + Top/Side extent → cylinder along Z or X/Y
    for fc in front.circles:
        c = _local_xy(front, fc.c)
        # height from top view extent at this x, or side at this y
        heights = []
        for tc in top.circles:
            tc_loc = _local_xy(top, tc.c)
            if _near(tc_loc[0], c[0], tol) and _near(tc.r, fc.r, tol):
                # circle in top → cylinder along Y
                heights.append(("y", tc_loc[1], fc.r))
        for sc in side.circles:
            sc_loc = _local_xy(side, sc.c)
            if _near(sc_loc[0], c[1], tol) and _near(sc.r, fc.r, tol):
                heights.append(("x", sc_loc[1], fc.r))
        # rectangle height from segs: z span in top at x≈c[0]
        z_vals = []
        for a, b in _view_segs_local(top):
            if _near(a[0], c[0], 2 * tol) or _near(b[0], c[0], 2 * tol):
                z_vals.extend([a[1], b[1]])
        if z_vals:
            z0, z1 = min(z_vals), max(z_vals)
            if z1 - z0 > tol:
                ax = gp_Ax2(gp_Pnt(c[0], c[1], z0), gp_Dir(0, 0, 1))
                cands.append(BRepPrimAPI_MakeCylinder(ax, fc.r, z1 - z0).Shape())

    if not cands:
        return None
    shape = cands[0]
    for other in cands[1:]:
        try:
            fuse = BRepAlgoAPI_Fuse(shape, other)
            fuse.Build()
            if fuse.IsDone():
                shape = fuse.Shape()
        except Exception:  # noqa: BLE001
            pass
    return shape


def _project_shape_edges(shape) -> dict[str, list[tuple[tuple[float, float], tuple[float, float]]]]:
    """Project OCC shape edges onto Front/Top/Side planes."""
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_EDGE
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.TopLoc import TopLoc_Location

    segs = {"front": [], "top": [], "side": []}
    exp = TopExp_Explorer(shape, TopAbs_EDGE)
    while exp.More():
        edge = exp.Current()
        curve, u0, u1 = BRep_Tool.Curve(edge)
        if curve is None:
            exp.Next()
            continue
        p0 = curve.Value(u0)
        p1 = curve.Value(u1)
        a = (p0.X(), p0.Y(), p0.Z())
        b = (p1.X(), p1.Y(), p1.Z())
        segs["front"].append(((a[0], a[1]), (b[0], b[1])))
        segs["top"].append(((a[0], a[2]), (b[0], b[2])))
        segs["side"].append(((a[1], a[2]), (b[1], b[2])))
        exp.Next()
    return segs


def backproject_score(
    shape,
    front: View2D,
    top: View2D,
    side: View2D,
    *,
    tol: float,
    side_swap: bool = True,
) -> float:
    """Fraction of input 2D segments covered by projected 3D edges (0..1)."""
    try:
        proj = _project_shape_edges(shape)
    except Exception:  # noqa: BLE001
        return 0.0
    scores = []
    for name, view, tgt_fn in (
        ("front", front, _view_segs_local),
        ("top", top, _view_segs_local),
        ("side", side, _side_segs_local if side_swap else _view_segs_local),
    ):
        tgt = tgt_fn(view)
        if not tgt:
            continue
        hit = 0
        for a, b in tgt:
            if _seg_covers(proj[name], a, b, 2 * tol):
                hit += 1
        scores.append(hit / max(len(tgt), 1))
    return float(sum(scores) / max(len(scores), 1)) if scores else 0.0


def _write_step(shape, path: Path) -> None:
    from occwl.io import save_step
    from occwl.solid import Solid
    from OCC.Core.TopoDS import TopoDS_Solid, topods
    from OCC.Core.TopAbs import TopAbs_SOLID
    from OCC.Core.TopExp import TopExp_Explorer

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    solid = None
    try:
        if shape.ShapeType() == TopAbs_SOLID:
            solid = Solid(topods.Solid(shape))
        else:
            exp = TopExp_Explorer(shape, TopAbs_SOLID)
            if exp.More():
                solid = Solid(topods.Solid(exp.Current()))
    except Exception:  # noqa: BLE001
        solid = None
    if solid is None:
        # wrap via compound writer
        from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs
        from OCC.Core.IFSelect import IFSelect_RetDone

        writer = STEPControl_Writer()
        writer.Transfer(shape, STEPControl_AsIs)
        status = writer.Write(str(path))
        if status != IFSelect_RetDone:
            raise RuntimeError(f"STEP write failed: {status}")
        return
    save_step([solid], path)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass
class ReconstructResult:
    ok: bool
    error: str = ""
    n_verts: int = 0
    n_edges: int = 0
    n_faces: int = 0
    score: float = 0.0
    method: str = ""


def reconstruct_views_to_step(
    views: dict[str, View2D],
    out_step: Path,
    *,
    tol: float | None = None,
    min_score: float = 0.15,
    sewing_tol: float = 1e-3,
) -> ReconstructResult:
    front0, top0, side0 = views["front"], views["top"], views["side"]
    spans = []
    for v in (front0, top0, side0):
        spans.append(float(np.max(v.bbox_max - v.bbox_min)))
    span = max(spans) if spans else 1.0
    tol = float(tol if tol is not None else max(1e-3, 1e-3 * span))

    # Layout / axis variants — pick highest back-projection score.
    view_variants: list[dict[str, View2D]] = [
        {"front": front0, "top": top0, "side": side0},
        {"front": top0, "top": front0, "side": side0},  # swapped front/top
    ]
    best_overall: Optional[ReconstructResult] = None
    best_shape = None

    for vv in view_variants:
        front, top, side = vv["front"], vv["top"], vv["side"]
        for side_swap in (True, False):
            verts = f_vr(front, top, side, tol=tol) if side_swap else f_vr_side_uv(
                front, top, side, tol=tol
            )
            if len(verts) < 4:
                continue
            edges = (
                f_ed(verts, front, top, side, tol=tol)
                if side_swap
                else f_ed_side_uv(verts, front, top, side, tol=tol)
            )
            loops = f_fa(verts, edges, tol=tol)
            shape, method = _build_solid_from_loops(verts, loops, sewing_tol=sewing_tol)
            if shape is None:
                box = _try_box_fallback(verts)
                if box is None:
                    continue
                shape, method = box, "bbox"
            score = backproject_score(
                shape, front, top, side, tol=tol, side_swap=side_swap
            )
            res = ReconstructResult(
                True,
                n_verts=len(verts),
                n_edges=len(edges),
                n_faces=len(loops),
                score=score,
                method=method,
            )
            if best_overall is None or score > best_overall.score:
                best_overall = res
                best_shape = shape

        cyl = _try_cylinders(front, top, side, tol=tol)
        if cyl is not None:
            score = backproject_score(cyl, front, top, side, tol=tol, side_swap=True)
            res = ReconstructResult(True, score=score, method="cylinder")
            if best_overall is None or score > best_overall.score:
                best_overall = res
                best_shape = cyl

    if best_overall is None or best_shape is None:
        return ReconstructResult(False, error="no_solid")
    if best_overall.score < 0.01 and best_overall.method != "bbox":
        best_overall.ok = False
        best_overall.error = "low_score"
        return best_overall
    try:
        _write_step(best_shape, out_step)
    except Exception as exc:  # noqa: BLE001
        best_overall.ok = False
        best_overall.error = f"write:{exc}"
        return best_overall
    return best_overall


def f_vr_side_uv(
    front: View2D, top: View2D, side: View2D, *, tol: float
) -> list[V3]:
    """Variant: Side local (u,v) mapped directly as (y,z)."""
    fv = _view_vertices(front, tol)
    tv = _view_vertices(top, tol)
    sv = _view_vertices(side, tol)  # (y,z)=(u,v)
    by_x_f: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for x, y in fv:
        by_x_f[int(round(x / tol))].append((x, y))
    by_x_t: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for x, z in tv:
        by_x_t[int(round(x / tol))].append((x, z))
    candidates: list[V3] = []
    seen: set[tuple[int, int, int]] = set()
    for kx, fpts in by_x_f.items():
        for kx2 in (kx - 1, kx, kx + 1):
            for x_f, y in fpts:
                for x_t, z in by_x_t.get(kx2, []):
                    if not _near(x_f, x_t, tol):
                        continue
                    x = 0.5 * (x_f + x_t)
                    if not _side_has(sv, y, z, tol):
                        continue
                    key = (
                        int(round(x / tol)),
                        int(round(y / tol)),
                        int(round(z / tol)),
                    )
                    if key in seen:
                        continue
                    seen.add(key)
                    candidates.append(V3(x, y, z))
    return candidates


def f_ed_side_uv(
    verts: Sequence[V3],
    front: View2D,
    top: View2D,
    side: View2D,
    *,
    tol: float,
    max_edges: int = 8000,
) -> list[tuple[int, int]]:
    fsegs = _view_segs_local(front)
    tsegs = _view_segs_local(top)
    ssegs = _view_segs_local(side)
    n = len(verts)
    edges: list[tuple[int, int]] = []
    pairs: list[tuple[float, int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            vi, vj = verts[i], verts[j]
            d = math.sqrt(
                (vi.x - vj.x) ** 2 + (vi.y - vj.y) ** 2 + (vi.z - vj.z) ** 2
            )
            if d < tol:
                continue
            pairs.append((d, i, j))
    pairs.sort(key=lambda t: t[0])
    for d, i, j in pairs:
        if len(edges) >= max_edges:
            break
        vi, vj = verts[i], verts[j]
        ok_f = _seg_covers(fsegs, (vi.x, vi.y), (vj.x, vj.y), tol) or (
            _near(vi.x, vj.x, tol) and _near(vi.y, vj.y, tol)
        )
        ok_t = _seg_covers(tsegs, (vi.x, vi.z), (vj.x, vj.z), tol) or (
            _near(vi.x, vj.x, tol) and _near(vi.z, vj.z, tol)
        )
        ok_s = _seg_covers(ssegs, (vi.y, vi.z), (vj.y, vj.z), tol) or (
            _near(vi.y, vj.y, tol) and _near(vi.z, vj.z, tol)
        )
        if ok_f and ok_t and ok_s:
            edges.append((i, j))
    return edges


def reconstruct_from_techdraw_paths(
    dataset_root: Path | str,
    *,
    dxf_rel: str,
    svg_rel: str = "",
    out_step: Path | str,
    tol: float | None = None,
    min_score: float = 0.15,
) -> ReconstructResult:
    views = load_labeled_views(Path(dataset_root), dxf_rel=dxf_rel, svg_rel=svg_rel)
    return reconstruct_views_to_step(
        views, Path(out_step), tol=tol, min_score=min_score
    )
