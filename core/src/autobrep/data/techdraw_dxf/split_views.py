"""Split a TechDraw sheet into 3 orthographic views via region-first layout analysis.

Pipeline (not primitive clustering):
  1. Filter / down-weight layout noise (short stubs, annotation layers).
  2. Build layout objects (bbox, center, length).
  3. Detect 3 view *regions* with recursive XY-Cut on projection gutters.
  4. Assign every primitive to a region by bbox overlap (not k-means).
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
      layout filter → XY-Cut view regions → bbox-overlap assignment.

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
            regions = _xy_cut_regions(objs, n_regions=k)
            if regions is not None and len(regions) >= k:
                labels = _assign_to_regions(objs, regions[:k])
                views = _labels_to_views_from_objs(objs, labels, n_views=n_views)
                if _nonempty_count(views) >= min(3, k):
                    return views
                _LOG.info("[split_into_views] XY-Cut assignment produced empty views")
        except Exception as exc:  # noqa: BLE001
            _LOG.info("[split_into_views] XY-Cut failed (%s); k-means fallback", exc)

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


def _xy_cut_regions(objs: list[LayoutObj], *, n_regions: int = 3) -> list[ViewRegion] | None:
    """
    Recursive XY-Cut: find largest projection gutter, split, recurse until n_regions.

    Uses signal (non-noise) objects to locate gutters; region boxes are tight
    bboxes of contained signal objects.
    """
    signal = _signal_objs(objs)
    if len(signal) < n_regions:
        return None

    def _bbox_of(group: list[LayoutObj]) -> ViewRegion:
        mn = np.stack([o.bbox_min for o in group]).min(axis=0).astype(np.float32)
        mx = np.stack([o.bbox_max for o in group]).max(axis=0).astype(np.float32)
        return ViewRegion(bbox_min=mn, bbox_max=mx)

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

    regions = [_bbox_of(g) for g in groups[:n_regions]]
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
