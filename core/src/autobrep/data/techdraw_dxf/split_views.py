"""Split a TechDraw sheet into 3 orthographic views via region-first layout analysis.

Pipeline (not primitive clustering):
  1. Filter / down-weight layout noise (short stubs, annotation layers).
  2. Build layout objects (bbox, center, length).
  3. Detect 3 view *regions* with constrained L-layout cuts on projection gutters
     (prefer: split the front/side band first, remainder = top — or the dual
     top/side row first). Recursive XY-Cut is only a fallback.
  4. Assign every primitive by hard cut planes (not k-means / soft bbox steal).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from autobrep.data.techdraw_dxf.extract import _collect_points
from autobrep.data.techdraw_dxf.schema import DxfIR, PrimIR, NUM_TD_VIEWS

_LOG = logging.getLogger(__name__)

# Layer name tokens treated as annotation / non-body for region detection.
_NOISE_LAYER_TOKENS = frozenset(
    {
        "DEFPOINTS",
        "ANNOTATION",
        "ANNO",
        "DIM",
        "DIMENSION",
        "DIMS",
        "TEXT",
        "TITLE",
        "BORDER",
        "FRAME",
        "HATCH",
        "TABLE",
        "BOM",
    }
)


@dataclass
class LayoutObj:
    prim: PrimIR
    index: int
    bbox_min: np.ndarray  # (2,)
    bbox_max: np.ndarray  # (2,)
    center: np.ndarray  # (2,)
    length: float
    weight: float  # contribution to projection density
    is_noise: bool


def prim_center(prim: PrimIR) -> np.ndarray:
    pts = _collect_points(prim)
    if not pts:
        return np.zeros(2, dtype=np.float32)
    return np.asarray(pts, dtype=np.float32).mean(axis=0)


def prim_bbox(prim: PrimIR) -> tuple[np.ndarray, np.ndarray]:
    pts = _collect_points(prim)
    if not pts:
        z = np.zeros(2, dtype=np.float32)
        return z, z + 1e-3
    arr = np.asarray(pts, dtype=np.float32)
    mn = arr.min(axis=0)
    mx = arr.max(axis=0)
    if np.allclose(mn, mx):
        mx = mn + 1e-3
    return mn, mx


def _prim_length(prim: PrimIR) -> float:
    if prim.type == "line":
        a = np.asarray(prim.params["start"][:2], dtype=np.float64)
        b = np.asarray(prim.params["end"][:2], dtype=np.float64)
        return float(np.linalg.norm(b - a))
    if prim.type == "circle":
        return 2.0 * np.pi * float(prim.params.get("radius", 0.0))
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


def merge_dxfir(parts: Sequence[DxfIR]) -> DxfIR:
    """
    Concatenate IR parts that live in a compatible sheet coordinate frame.

    When DXF and SVG disagree on bbox (common: SVG uses a different y-up /
    pixel frame), keep the densest compatible subset — prefer the part with
    more primitives when IoU of sheet bboxes is near zero.
    """
    parts = [p for p in parts if int(p.n_prims) > 0]
    if not parts:
        return DxfIR(
            n_prims=0,
            prims=[],
            bbox_min=np.zeros(2, dtype=np.float32),
            bbox_max=np.ones(2, dtype=np.float32),
        )
    if len(parts) == 1:
        return _dxfir_from_prims(list(parts[0].prims[: int(parts[0].n_prims)]))

    def _iou(a: DxfIR, b: DxfIR) -> float:
        ax0, ay0 = map(float, a.bbox_min)
        ax1, ay1 = map(float, a.bbox_max)
        bx0, by0 = map(float, b.bbox_min)
        bx1, by1 = map(float, b.bbox_max)
        ix0, iy0 = max(ax0, bx0), max(ay0, by0)
        ix1, iy1 = min(ax1, bx1), min(ay1, by1)
        inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
        aa = max((ax1 - ax0) * (ay1 - ay0), 1e-12)
        ba = max((bx1 - bx0) * (by1 - by0), 1e-12)
        return inter / (aa + ba - inter + 1e-12)

    ordered = sorted(parts, key=lambda p: int(p.n_prims), reverse=True)
    keep = [ordered[0]]
    for p in ordered[1:]:
        ok = False
        for q in keep:
            iou = _iou(p, q)
            ps = np.asarray(p.bbox_max, dtype=np.float64) - np.asarray(
                p.bbox_min, dtype=np.float64
            )
            qs = np.asarray(q.bbox_max, dtype=np.float64) - np.asarray(
                q.bbox_min, dtype=np.float64
            )
            scale = float(np.max(ps) / max(float(np.max(qs)), 1e-6))
            # Strong overlap → same sheet
            if iou >= 0.2:
                ok = True
                break
            # Nested / similar extent
            pc = 0.5 * (np.asarray(p.bbox_min) + np.asarray(p.bbox_max))
            qc = 0.5 * (np.asarray(q.bbox_min) + np.asarray(q.bbox_max))
            if (
                iou >= 0.08
                and 0.5 <= scale <= 2.0
                and float(np.linalg.norm(pc - qc)) < 0.35 * float(np.max(qs))
            ):
                ok = True
                break
        if ok:
            keep.append(p)
        else:
            _LOG.info(
                "[merge_dxfir] drop incompatible IR n=%s bbox=[%s,%s]-[%s,%s]",
                int(p.n_prims),
                *map(float, p.bbox_min),
                *map(float, p.bbox_max),
            )

    prims: list[PrimIR] = []
    for p in keep:
        prims.extend(list(p.prims[: int(p.n_prims)]))
    return _dxfir_from_prims(prims)


def split_into_views(
    dxfir: DxfIR,
    *,
    n_views: int = NUM_TD_VIEWS,
    use_histogram: bool = True,
) -> list[DxfIR]:
    """
    Partition primitives into ``n_views`` sheet views.

    Preferred path (``use_histogram=True``):
      layout filter → L-layout gutter cuts → hard plane assignment.

    ``use_histogram=False`` falls back to pure k-means on centers (legacy).
    """
    prims = list(dxfir.prims[: int(dxfir.n_prims)])
    empty = [_empty_view() for _ in range(n_views)]
    if not prims:
        return empty

    k = min(n_views, len(prims))
    if k == 1:
        return [_dxfir_from_prims(prims)] + empty[1:]

    objs = _build_layout_objects(prims)
    if use_histogram and k >= 3 and len(prims) >= 3:
        try:
            plan = _l_layout_plan(objs, n_regions=k)
            if plan is not None:
                labels = _assign_by_plan(objs, plan)
                views = _labels_to_views_from_objs(objs, labels, n_views=n_views)
                if _nonempty_count(views) >= min(3, k):
                    return views
                _LOG.info("[split_into_views] L-layout assignment produced empty views")
            regions = _xy_cut_regions(objs, n_regions=k)
            if regions is not None and len(regions) >= k:
                labels = _assign_to_regions(objs, regions[:k])
                views = _labels_to_views_from_objs(objs, labels, n_views=n_views)
                if _nonempty_count(views) >= min(3, k):
                    return views
                _LOG.info("[split_into_views] XY-Cut assignment produced empty views")
        except Exception as exc:  # noqa: BLE001
            _LOG.info("[split_into_views] region split failed (%s); k-means fallback", exc)

    centers = np.stack([o.center for o in objs], axis=0)
    labels = _cluster_labels(centers, k=k)
    return _labels_to_views_from_objs(objs, labels, n_views=n_views)


def _empty_view() -> DxfIR:
    return DxfIR(
        n_prims=0,
        prims=[],
        bbox_min=np.zeros(2, dtype=np.float32),
        bbox_max=np.ones(2, dtype=np.float32),
    )


def _build_layout_objects(prims: list[PrimIR]) -> list[LayoutObj]:
    raw: list[LayoutObj] = []
    for i, p in enumerate(prims):
        mn, mx = prim_bbox(p)
        length = max(_prim_length(p), 1e-6)
        raw.append(
            LayoutObj(
                prim=p,
                index=i,
                bbox_min=mn.astype(np.float32),
                bbox_max=mx.astype(np.float32),
                center=((mn + mx) * 0.5).astype(np.float32),
                length=float(length),
                weight=1.0,
                is_noise=False,
            )
        )
    if not raw:
        return raw

    sheet_min = np.stack([o.bbox_min for o in raw]).min(axis=0)
    sheet_max = np.stack([o.bbox_max for o in raw]).max(axis=0)
    span = float(np.max(sheet_max - sheet_min)) + 1e-6
    # Relative short stubs / annotation layers → noise for region detection only.
    short_thr = max(1e-3, 0.008 * span)
    out: list[LayoutObj] = []
    for o in raw:
        layer = str((o.prim.params or {}).get("layer", "")).upper()
        layer_noise = bool(layer) and any(tok in layer for tok in _NOISE_LAYER_TOKENS)
        short = o.length < short_thr and o.prim.type in {"line", "lwpolyline", "other"}
        is_noise = layer_noise or short
        # Circles/arcs keep full weight (structure); noise lines get tiny weight.
        if is_noise:
            w = 0.05 * o.length
        elif o.prim.type in {"circle", "arc", "ellipse"}:
            w = max(o.length, 0.02 * span)
        else:
            w = o.length
        out.append(
            LayoutObj(
                prim=o.prim,
                index=o.index,
                bbox_min=o.bbox_min,
                bbox_max=o.bbox_max,
                center=o.center,
                length=o.length,
                weight=float(max(w, 1e-6)),
                is_noise=is_noise,
            )
        )
    return out


def _signal_objs(objs: list[LayoutObj]) -> list[LayoutObj]:
    sig = [o for o in objs if not o.is_noise]
    return sig if len(sig) >= 3 else list(objs)


def _empty_gaps_1d(
    values: np.ndarray,
    weights: np.ndarray,
    *,
    n_bins: int = 96,
    min_frac: float = 0.05,
) -> list[tuple[float, float, float]]:
    """Return gutters as ``(width, mid, score)`` widest-first. score≈width*isolation."""
    if len(values) < 2:
        return []
    lo, hi = float(values.min()), float(values.max())
    span = hi - lo
    if span < 1e-9:
        return []
    hist, edges = np.histogram(values, bins=n_bins, range=(lo, hi), weights=weights)
    thr = max(float(weights.sum()) * 0.01, float(np.median(weights)) * 0.5)
    gaps: list[tuple[float, float, float]] = []
    i = 0
    while i < n_bins:
        if hist[i] <= thr:
            j = i
            while j < n_bins and hist[j] <= thr:
                j += 1
            if i > 0 and j < n_bins and hist[:i].sum() > 0 and hist[j:].sum() > 0:
                width = float(edges[j] - edges[i])
                if width >= min_frac * span:
                    mid = 0.5 * (float(edges[i]) + float(edges[j]))
                    left_mass = float(hist[:i].sum())
                    right_mass = float(hist[j:].sum())
                    bal = min(left_mass, right_mass) / max(left_mass, right_mass, 1e-6)
                    score = (width / span) * (0.5 + 0.5 * bal)
                    gaps.append((width, mid, score))
            i = j
        else:
            i += 1
    gaps.sort(key=lambda t: -t[2])
    return gaps


@dataclass
class ViewRegion:
    bbox_min: np.ndarray
    bbox_max: np.ndarray

    @property
    def center(self) -> np.ndarray:
        return 0.5 * (self.bbox_min + self.bbox_max)


@dataclass
class LLayoutPlan:
    """Constrained 3-view sheet cut (no recursive over-splitting of one view)."""

    mode: str  # "xy" | "yx"
    x_cut: float
    y_cut: float
    regions: list[ViewRegion]


def _bbox_of_objs(group: list[LayoutObj]) -> ViewRegion:
    mn = np.stack([o.bbox_min for o in group]).min(axis=0).astype(np.float32)
    mx = np.stack([o.bbox_max for o in group]).max(axis=0).astype(np.float32)
    return ViewRegion(bbox_min=mn, bbox_max=mx)


def _best_gap(
    group: list[LayoutObj],
    axis: int,
    *,
    min_frac: float = 0.05,
) -> tuple[float, float, float] | None:
    """Return ``(width, mid, score)`` of best gutter on axis, or None."""
    if len(group) < 2:
        return None
    centers = np.stack([o.center for o in group])
    weights = np.asarray([o.weight for o in group], dtype=np.float64)
    gaps = _empty_gaps_1d(centers[:, axis], weights, min_frac=min_frac)
    return gaps[0] if gaps else None


def _extent_align_score(a: ViewRegion, b: ViewRegion, axis: int) -> float:
    """1 if two regions share nearly the same span on ``axis`` (width or height)."""
    a0, a1 = float(a.bbox_min[axis]), float(a.bbox_max[axis])
    b0, b1 = float(b.bbox_min[axis]), float(b.bbox_max[axis])
    span = max(a1 - a0, b1 - b0, 1e-6)
    # center shift + length mismatch, normalized
    mid_diff = abs(0.5 * (a0 + a1) - 0.5 * (b0 + b1)) / span
    len_diff = abs((a1 - a0) - (b1 - b0)) / span
    return float(np.clip(1.0 - 0.5 * mid_diff - 0.5 * len_diff, 0.0, 1.0))


def _plan_from_groups(
    groups: list[list[LayoutObj]],
    *,
    mode: str,
    x_cut: float,
    y_cut: float,
    gutter_score: float,
) -> tuple[LLayoutPlan, float] | None:
    nonempty = [g for g in groups if g]
    if len(nonempty) < 3:
        return None
    # Keep three densest if somehow >3
    nonempty = sorted(nonempty, key=len, reverse=True)[:3]
    regions = [_bbox_of_objs(g) for g in nonempty]
    # Alignment: one pair shares width (axis=0), one pair shares height (axis=1).
    # Typical L sheet: front↔top same width; top↔side same height.
    best_align = 0.0
    for i in range(3):
        for j in range(i + 1, 3):
            for k in range(3):
                if k == i or k == j:
                    continue
                # (i,j) share width; (j,k) share height — or swap
                s1 = _extent_align_score(regions[i], regions[j], 0) + _extent_align_score(
                    regions[j], regions[k], 1
                )
                s2 = _extent_align_score(regions[i], regions[j], 1) + _extent_align_score(
                    regions[j], regions[k], 0
                )
                best_align = max(best_align, s1, s2)
    sizes = sorted(len(g) for g in nonempty)
    bal = sizes[0] / max(sizes[-1], 1)
    # Prefer wide gutters, orthographic alignment, and not one tiny leftover.
    score = gutter_score * (0.35 + 0.35 * best_align + 0.30 * bal)
    # Light pad so borderline centers still fall inside their region box (viz only);
    # assignment uses hard cuts, not these boxes.
    for r in regions:
        span = np.maximum(r.bbox_max - r.bbox_min, 1e-3)
        pad = 0.01 * span
        r.bbox_min = (r.bbox_min - pad).astype(np.float32)
        r.bbox_max = (r.bbox_max + pad).astype(np.float32)
    plan = LLayoutPlan(mode=mode, x_cut=float(x_cut), y_cut=float(y_cut), regions=regions)
    return plan, float(score)


def _l_layout_plan(objs: list[LayoutObj], *, n_regions: int = 3) -> LLayoutPlan | None:
    """
    Constrained L-layout split for 3 orthographic views.

    Two candidate cut orders (pick higher score):

    * **xy** — first split left/right (main column vs side), then split the
      left column into front/top. Matches: 「先主/侧，剩余俯视」.
    * **yx** — first split high/low (front vs top+side row), then split the
      multi-view row into top/side. Matches sheets where top & side share height
      (e.g. train/000008).

    Recursive XY-Cut is avoided as the primary path because it may bisect the
    side view's internal gaps instead of separating the left column.
    """
    signal = _signal_objs(objs)
    if len(signal) < n_regions:
        return None

    candidates: list[tuple[LLayoutPlan, float]] = []

    # --- Path xy: main|side first, then front/top on the left column ---
    xg = _best_gap(signal, 0)
    if xg is not None:
        _, x_mid, x_score = xg
        left = [o for o in signal if o.center[0] < x_mid]
        right = [o for o in signal if o.center[0] >= x_mid]
        # Split the column that still stacks two views (usually left).
        for primary, other in ((left, right), (right, left)):
            if len(primary) < 2 or not other:
                continue
            yg = _best_gap(primary, 1)
            if yg is None:
                continue
            _, y_mid, y_score = yg
            hi = [o for o in primary if o.center[1] >= y_mid]
            lo = [o for o in primary if o.center[1] < y_mid]
            if not hi or not lo:
                continue
            got = _plan_from_groups(
                [hi, lo, other],
                mode="xy",
                x_cut=x_mid,
                y_cut=y_mid,
                gutter_score=x_score + y_score,
            )
            if got is not None:
                candidates.append(got)

    # --- Path yx: front vs (top|side) row first, then top/side ---
    yg = _best_gap(signal, 1)
    if yg is not None:
        _, y_mid, y_score = yg
        high = [o for o in signal if o.center[1] >= y_mid]
        low = [o for o in signal if o.center[1] < y_mid]
        for primary, other in ((low, high), (high, low)):
            if len(primary) < 2 or not other:
                continue
            xg2 = _best_gap(primary, 0)
            if xg2 is None:
                continue
            _, x_mid, x_score = xg2
            left = [o for o in primary if o.center[0] < x_mid]
            right = [o for o in primary if o.center[0] >= x_mid]
            if not left or not right:
                continue
            got = _plan_from_groups(
                [left, right, other],
                mode="yx",
                x_cut=x_mid,
                y_cut=y_mid,
                gutter_score=y_score + x_score,
            )
            if got is not None:
                candidates.append(got)

    if not candidates:
        return None
    candidates.sort(key=lambda t: -t[1])
    return candidates[0][0]


def _assign_by_plan(objs: list[LayoutObj], plan: LLayoutPlan) -> np.ndarray:
    """
    Hard plane assignment from L-cuts.

    Each primitive is placed by (x_cut, y_cut) quadrant, then matched to the
    plan region whose center lies in that quadrant. Prevents a long border
    line of the front view from being stolen by a tight top-view bbox.
    """
    x_cut, y_cut = plan.x_cut, plan.y_cut
    reg_centers = [r.center for r in plan.regions]
    labels = np.zeros(len(objs), dtype=np.int64)
    for i, o in enumerate(objs):
        cx, cy = float(o.center[0]), float(o.center[1])
        left = cx < x_cut
        high = cy >= y_cut
        cands: list[int] = []
        for k, rc in enumerate(reg_centers):
            r_left = float(rc[0]) < x_cut
            r_high = float(rc[1]) >= y_cut
            if r_left == left and r_high == high:
                cands.append(k)
        if not cands:
            dists = [float(np.linalg.norm(o.center - rc)) for rc in reg_centers]
            labels[i] = int(np.argmin(dists))
        elif len(cands) == 1:
            labels[i] = cands[0]
        else:
            dists = [float(np.linalg.norm(o.center - reg_centers[k])) for k in cands]
            labels[i] = cands[int(np.argmin(dists))]
    return labels


def _xy_cut_regions(objs: list[LayoutObj], *, n_regions: int = 3) -> list[ViewRegion] | None:
    """
    Recursive XY-Cut fallback: find largest projection gutter, split, recurse.

    Uses signal (non-noise) objects to locate gutters; region boxes are tight
    bboxes of contained signal objects.

    Prefer :func:`_l_layout_plan` for 3-view sheets — recursive cuts can bisect
    one view's internal gaps (e.g. side) instead of separating front/top.
    """
    signal = _signal_objs(objs)
    if len(signal) < n_regions:
        return None

    def _split(group: list[LayoutObj]) -> tuple[list[LayoutObj], list[LayoutObj], float] | None:
        if len(group) < 2:
            return None
        centers = np.stack([o.center for o in group])
        weights = np.asarray([o.weight for o in group], dtype=np.float64)
        best: tuple[list[LayoutObj], list[LayoutObj], float] | None = None
        for axis in (0, 1):
            gaps = _empty_gaps_1d(centers[:, axis], weights)
            if not gaps:
                continue
            # try top-2 gaps on this axis
            for width, mid, score in gaps[:2]:
                left = [o for o in group if o.center[axis] < mid]
                right = [o for o in group if o.center[axis] >= mid]
                if not left or not right:
                    continue
                # prefer balanced nonempty splits with wide gutters
                bal = min(len(left), len(right)) / max(len(left), len(right))
                split_score = score * (0.4 + 0.6 * bal)
                if best is None or split_score > best[2]:
                    best = (left, right, split_score)
        return best

    # Start with one group; repeatedly split the largest group until n_regions.
    groups: list[list[LayoutObj]] = [list(signal)]
    while len(groups) < n_regions:
        # Split the group with the best available cut
        cand_idx = -1
        cand_split: tuple[list[LayoutObj], list[LayoutObj], float] | None = None
        for i, g in enumerate(groups):
            sp = _split(g)
            if sp is None:
                continue
            if cand_split is None or sp[2] > cand_split[2]:
                cand_split = sp
                cand_idx = i
        if cand_split is None or cand_idx < 0:
            break
        left, right, _ = cand_split
        groups.pop(cand_idx)
        groups.append(left)
        groups.append(right)

    if len(groups) < n_regions:
        # Fallback: single hard L-cut (one x + one y gutter) → 4 quads, drop emptiest
        centers = np.stack([o.center for o in signal])
        weights = np.asarray([o.weight for o in signal], dtype=np.float64)
        x_gaps = _empty_gaps_1d(centers[:, 0], weights)
        y_gaps = _empty_gaps_1d(centers[:, 1], weights)
        if not x_gaps or not y_gaps:
            return None
        x_cut = x_gaps[0][1]
        y_cut = y_gaps[0][1]
        quads: list[list[LayoutObj]] = [[] for _ in range(4)]
        for o in signal:
            left = o.center[0] < x_cut
            top = o.center[1] >= y_cut
            if left and top:
                quads[0].append(o)
            elif left and not top:
                quads[1].append(o)
            elif (not left) and top:
                quads[2].append(o)
            else:
                quads[3].append(o)
        nonempty = [q for q in quads if q]
        if len(nonempty) < n_regions:
            return None
        # Prefer keeping densest n_regions; if 4, drop smallest (usually empty BR)
        nonempty.sort(key=len, reverse=True)
        groups = nonempty[:n_regions]

    regions = [_bbox_of_objs(g) for g in groups[:n_regions]]
    # Expand boxes slightly so borderline prims still overlap their region.
    for r in regions:
        span = np.maximum(r.bbox_max - r.bbox_min, 1e-3)
        pad = 0.02 * span
        r.bbox_min = (r.bbox_min - pad).astype(np.float32)
        r.bbox_max = (r.bbox_max + pad).astype(np.float32)
    return regions


def _overlap_ratio(mn: np.ndarray, mx: np.ndarray, region: ViewRegion) -> float:
    """Intersection area / prim bbox area."""
    ix0 = max(float(mn[0]), float(region.bbox_min[0]))
    iy0 = max(float(mn[1]), float(region.bbox_min[1]))
    ix1 = min(float(mx[0]), float(region.bbox_max[0]))
    iy1 = min(float(mx[1]), float(region.bbox_max[1]))
    iw = max(0.0, ix1 - ix0)
    ih = max(0.0, iy1 - iy0)
    inter = iw * ih
    area = max(float((mx[0] - mn[0]) * (mx[1] - mn[1])), 1e-12)
    return inter / area


def _center_inside(center: np.ndarray, region: ViewRegion) -> bool:
    return bool(
        region.bbox_min[0] <= center[0] <= region.bbox_max[0]
        and region.bbox_min[1] <= center[1] <= region.bbox_max[1]
    )


def _assign_to_regions(objs: list[LayoutObj], regions: list[ViewRegion]) -> np.ndarray:
    """Assign each primitive to the region with max bbox-overlap (ties: center-in / nearest)."""
    labels = np.zeros(len(objs), dtype=np.int64)
    for i, o in enumerate(objs):
        scores = []
        for k, r in enumerate(regions):
            ov = _overlap_ratio(o.bbox_min, o.bbox_max, r)
            bonus = 0.15 if _center_inside(o.center, r) else 0.0
            # distance penalty if no overlap
            rc = r.center
            dist = float(np.linalg.norm(o.center - rc))
            diag = float(np.linalg.norm(r.bbox_max - r.bbox_min)) + 1e-6
            scores.append(ov + bonus - 0.05 * (dist / diag))
        labels[i] = int(np.argmax(scores))
    return labels


def _labels_to_views_from_objs(
    objs: list[LayoutObj],
    labels: np.ndarray,
    *,
    n_views: int,
) -> list[DxfIR]:
    k = int(labels.max()) + 1 if len(labels) else 0
    if k <= 0:
        return [_empty_view() for _ in range(n_views)]
    centers = np.stack([o.center for o in objs])
    centroids = np.stack(
        [
            centers[labels == i].mean(axis=0)
            if (labels == i).any()
            else np.zeros(2, dtype=np.float32)
            for i in range(k)
        ],
        axis=0,
    )
    # CAD y-up: high-y first (sheet "top" in data coords often = front in L-layout bottom after invert viz)
    order = sorted(
        range(k),
        key=lambda i: (-float(centroids[i, 1]), float(centroids[i, 0])),
    )
    remap = {old: new for new, old in enumerate(order)}
    buckets: list[list[PrimIR]] = [[] for _ in range(n_views)]
    for o, lab in zip(objs, labels):
        slot = remap[int(lab)]
        if slot < n_views:
            buckets[slot].append(o.prim)
    return [_dxfir_from_prims(b) for b in buckets]


def _nonempty_count(views: list[DxfIR]) -> int:
    return sum(1 for v in views if int(v.n_prims) > 0)


def _dxfir_from_prims(prims: list[PrimIR]) -> DxfIR:
    if not prims:
        return _empty_view()
    pts = []
    for prim in prims:
        pts.extend(_collect_points(prim))
    arr = np.asarray(pts, dtype=np.float32) if pts else np.zeros((1, 2), np.float32)
    bbox_min = arr.min(axis=0).astype(np.float32)
    bbox_max = arr.max(axis=0).astype(np.float32)
    if np.allclose(bbox_min, bbox_max):
        bbox_max = bbox_min + 1.0
    return DxfIR(n_prims=len(prims), prims=prims, bbox_min=bbox_min, bbox_max=bbox_max)


def _cluster_labels(
    centers: np.ndarray,
    *,
    k: int,
    init_means: np.ndarray | None = None,
) -> np.ndarray:
    """Legacy k-means fallback."""
    n = centers.shape[0]
    if init_means is not None and len(init_means) == k:
        means = np.asarray(init_means, dtype=np.float32).copy()
    else:
        score = centers[:, 0] + centers[:, 1]
        qs = np.linspace(0, n - 1, num=k).astype(int)
        order = np.argsort(score)
        means = centers[order[qs]].copy()
    labels = np.zeros(n, dtype=np.int64)
    for _ in range(16):
        d = ((centers[:, None, :] - means[None, :, :]) ** 2).sum(-1)
        labels = d.argmin(axis=1)
        new_means = means.copy()
        for i in range(k):
            m = labels == i
            if m.any():
                new_means[i] = centers[m].mean(axis=0)
        if np.allclose(new_means, means):
            break
        means = new_means
    return labels
