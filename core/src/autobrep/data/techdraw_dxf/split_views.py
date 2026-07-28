"""Split a TechDraw sheet (DXF/SVG prims) into 3 orthographic view groups."""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np

from autobrep.data.techdraw_dxf.extract import _collect_points
from autobrep.data.techdraw_dxf.schema import DxfIR, PrimIR, NUM_TD_VIEWS

_LOG = logging.getLogger(__name__)


def prim_center(prim: PrimIR) -> np.ndarray:
    pts = _collect_points(prim)
    if not pts:
        return np.zeros(2, dtype=np.float32)
    return np.asarray(pts, dtype=np.float32).mean(axis=0)


def merge_dxfir(parts: Sequence[DxfIR]) -> DxfIR:
    prims: list[PrimIR] = []
    for p in parts:
        prims.extend(list(p.prims[: int(p.n_prims)]))
    if not prims:
        return DxfIR(
            n_prims=0,
            prims=[],
            bbox_min=np.zeros(2, dtype=np.float32),
            bbox_max=np.ones(2, dtype=np.float32),
        )
    pts = []
    for prim in prims:
        pts.extend(_collect_points(prim))
    arr = np.asarray(pts, dtype=np.float32) if pts else np.zeros((1, 2), np.float32)
    return DxfIR(
        n_prims=len(prims),
        prims=prims,
        bbox_min=arr.min(axis=0).astype(np.float32),
        bbox_max=arr.max(axis=0).astype(np.float32),
    )


def split_into_views(
    dxfir: DxfIR,
    *,
    n_views: int = NUM_TD_VIEWS,
    use_histogram: bool = True,
) -> list[DxfIR]:
    """
    Spatially cluster primitives into ``n_views`` groups (TechDraw sheet layout).

    Default pipeline (``use_histogram=True``):
      1. xy projection histograms → valley seeds for coarse regions
      2. within-seed k-means refine
      3. name slots as top-left / bottom-left / top-right via (-y, x) centroid sort
      4. width/height projection-consistency check; on failure fall back to pure k-means

    Ordering: sort cluster centroids by (-y, x) so top-left-ish views come first.
    """
    prims = list(dxfir.prims[: int(dxfir.n_prims)])
    empty = [
        DxfIR(
            n_prims=0,
            prims=[],
            bbox_min=np.zeros(2, dtype=np.float32),
            bbox_max=np.ones(2, dtype=np.float32),
        )
        for _ in range(n_views)
    ]
    if not prims:
        return empty

    centers = np.stack([prim_center(p) for p in prims], axis=0).astype(np.float32)
    k = min(n_views, len(prims))
    if k == 1:
        return [_dxfir_from_prims(prims)] + empty[1:]

    if use_histogram and k >= 3 and len(prims) >= 3:
        try:
            labels = _hist_then_kmeans(centers, k=k)
            views = _labels_to_views(prims, centers, labels, n_views=n_views)
            if _projection_consistent(views) and _nonempty_count(views) >= min(3, k):
                return views
            _LOG.info(
                "[split_into_views] projection consistency failed; falling back to k-means"
            )
        except Exception as exc:  # noqa: BLE001
            _LOG.info("[split_into_views] histogram split failed (%s); k-means fallback", exc)

    labels = _cluster_labels(centers, k=k)
    return _labels_to_views(prims, centers, labels, n_views=n_views)


def _nonempty_count(views: list[DxfIR]) -> int:
    return sum(1 for v in views if int(v.n_prims) > 0)


def _projection_consistent(views: list[DxfIR], *, rel_tol: float = 0.35) -> bool:
    """
    Check standard 3-view sheet: slots 0=TL (front), 1=BL (top), 2=TR (side).

    Front width ≈ top width; front height ≈ side height (relative).
    """
    if len(views) < 3:
        return True
    sizes = []
    for v in views[:3]:
        if int(v.n_prims) <= 0:
            return False
        wh = np.asarray(v.bbox_max, dtype=np.float64) - np.asarray(
            v.bbox_min, dtype=np.float64
        )
        sizes.append(wh)
    front_w, front_h = float(sizes[0][0]), float(sizes[0][1])
    top_w, top_h = float(sizes[1][0]), float(sizes[1][1])
    side_w, side_h = float(sizes[2][0]), float(sizes[2][1])
    if front_w < 1e-8 or front_h < 1e-8:
        return False

    def _close(a: float, b: float) -> bool:
        m = max(abs(a), abs(b), 1e-8)
        return abs(a - b) / m <= rel_tol

    # front.width ~ top.width; front.height ~ side.height
    ok_w = _close(front_w, top_w)
    ok_h = _close(front_h, side_h)
    # soft: also allow swapped naming (front vs side) if one pair matches strongly
    if ok_w and ok_h:
        return True
    if _close(front_w, side_w) and _close(front_h, top_h):
        return True
    return ok_w or ok_h


def _labels_to_views(
    prims: list[PrimIR],
    centers: np.ndarray,
    labels: np.ndarray,
    *,
    n_views: int,
) -> list[DxfIR]:
    k = int(labels.max()) + 1 if len(labels) else 0
    if k <= 0:
        return [
            DxfIR(
                n_prims=0,
                prims=[],
                bbox_min=np.zeros(2, dtype=np.float32),
                bbox_max=np.ones(2, dtype=np.float32),
            )
            for _ in range(n_views)
        ]
    centroids = np.stack(
        [
            centers[labels == i].mean(axis=0)
            if (labels == i).any()
            else np.zeros(2, dtype=np.float32)
            for i in range(k)
        ],
        axis=0,
    )
    order = sorted(
        range(k),
        key=lambda i: (-float(centroids[i, 1]), float(centroids[i, 0])),
    )
    remap = {old: new for new, old in enumerate(order)}
    buckets: list[list[PrimIR]] = [[] for _ in range(n_views)]
    for prim, lab in zip(prims, labels):
        slot = remap[int(lab)]
        if slot < n_views:
            buckets[slot].append(prim)
    return [_dxfir_from_prims(b) for b in buckets]


def _hist_valleys(values: np.ndarray, *, n_bins: int = 64) -> list[float]:
    """Return valley x-positions (in data coords) of a 1D histogram."""
    if len(values) < 2:
        return []
    lo, hi = float(values.min()), float(values.max())
    if hi - lo < 1e-9:
        return []
    hist, edges = np.histogram(values, bins=n_bins, range=(lo, hi))
    # smooth
    kernel = np.array([1, 2, 3, 2, 1], dtype=np.float64)
    kernel /= kernel.sum()
    pad = len(kernel) // 2
    sm = np.convolve(hist.astype(np.float64), kernel, mode="same")
    # find local minima in interior
    valleys: list[float] = []
    for i in range(pad, len(sm) - pad):
        if sm[i] <= sm[i - 1] and sm[i] <= sm[i + 1] and sm[i] < sm.max() * 0.35:
            valleys.append(0.5 * (float(edges[i]) + float(edges[i + 1])))
    return valleys


def _hist_then_kmeans(centers: np.ndarray, *, k: int) -> np.ndarray:
    """
    Coarse partition via x/y histogram valleys, then k-means refine with seeded means.
    """
    xs = centers[:, 0]
    ys = centers[:, 1]
    x_valleys = _hist_valleys(xs)
    y_valleys = _hist_valleys(ys)

    # Prefer one vertical + one horizontal cut → 4 quadrants, keep 3 densest / named later
    x_cut = None
    y_cut = None
    if x_valleys:
        # pick valley closest to median
        med_x = float(np.median(xs))
        x_cut = min(x_valleys, key=lambda v: abs(v - med_x))
    if y_valleys:
        med_y = float(np.median(ys))
        y_cut = min(y_valleys, key=lambda v: abs(v - med_y))

    if x_cut is None and y_cut is None:
        return _cluster_labels(centers, k=k)

    # quadrant / strip labels as seeds
    if x_cut is not None and y_cut is not None:
        coarse = np.zeros(len(centers), dtype=np.int64)
        left = xs < x_cut
        top = ys >= y_cut
        # 0 TL, 1 BL, 2 TR, 3 BR
        coarse[left & top] = 0
        coarse[left & ~top] = 1
        coarse[~left & top] = 2
        coarse[~left & ~top] = 3
        # Drop emptiest quadrant so we keep 3 (or merge BR into nearest)
        counts = np.bincount(coarse, minlength=4)
        drop = int(np.argmin(counts))
        keep = [i for i in range(4) if i != drop][:k]
        remap = {old: i for i, old in enumerate(keep)}
        # reassign dropped to nearest kept centroid
        kept_means = []
        for old in keep:
            m = centers[coarse == old]
            kept_means.append(m.mean(axis=0) if len(m) else centers.mean(axis=0))
        kept_means_a = np.stack(kept_means, axis=0)
        labels = np.zeros(len(centers), dtype=np.int64)
        for i, c in enumerate(centers):
            old = int(coarse[i])
            if old in remap:
                labels[i] = remap[old]
            else:
                labels[i] = int(((kept_means_a - c) ** 2).sum(-1).argmin())
        seeds = kept_means_a.copy()
    else:
        cut = x_cut if x_cut is not None else y_cut
        axis = 0 if x_cut is not None else 1
        coarse = (centers[:, axis] >= cut).astype(np.int64)
        # split denser side further by the other axis median
        denser = int(np.bincount(coarse, minlength=2).argmax())
        other = 1 - denser
        sub = centers[coarse == denser]
        if len(sub) < 2:
            return _cluster_labels(centers, k=k)
        med = float(np.median(sub[:, 1 - axis]))
        labels = np.zeros(len(centers), dtype=np.int64)
        labels[coarse == other] = 0
        mask = coarse == denser
        labels[mask] = 1 + (centers[mask, 1 - axis] >= med).astype(np.int64)
        # ensure k labels
        uniq = sorted(set(int(x) for x in labels.tolist()))
        if len(uniq) < k:
            return _cluster_labels(centers, k=k)
        remap = {old: i for i, old in enumerate(uniq[:k])}
        labels = np.asarray([remap.get(int(x), k - 1) for x in labels], dtype=np.int64)
        seeds = np.stack(
            [centers[labels == i].mean(axis=0) for i in range(k)], axis=0
        )

    return _cluster_labels(centers, k=k, init_means=seeds)


def _dxfir_from_prims(prims: list[PrimIR]) -> DxfIR:
    if not prims:
        return DxfIR(
            n_prims=0,
            prims=[],
            bbox_min=np.zeros(2, dtype=np.float32),
            bbox_max=np.ones(2, dtype=np.float32),
        )
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
    """Lightweight k-means (no sklearn dependency in training path)."""
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
