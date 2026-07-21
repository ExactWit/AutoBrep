"""Split a TechDraw sheet (DXF/SVG prims) into 3 orthographic view groups."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from autobrep.data.techdraw_dxf.extract import _collect_points
from autobrep.data.techdraw_dxf.schema import DxfIR, PrimIR, NUM_TD_VIEWS


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
) -> list[DxfIR]:
    """
    Spatially cluster primitives into ``n_views`` groups (TechDraw sheet layout).

    Ordering: sort cluster centroids by (-y, x) so top-left-ish views come first
    (stable across samples; not aligned to render shading styles).
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

    labels = _cluster_labels(centers, k=k)
    # map cluster id → ordered view slot
    centroids = np.stack(
        [centers[labels == i].mean(axis=0) for i in range(k)], axis=0
    )
    order = sorted(
        range(k),
        key=lambda i: (-float(centroids[i, 1]), float(centroids[i, 0])),
    )
    remap = {old: new for new, old in enumerate(order)}

    buckets: list[list[PrimIR]] = [[] for _ in range(n_views)]
    for prim, lab in zip(prims, labels):
        buckets[remap[int(lab)]].append(prim)

    out: list[DxfIR] = []
    for b in buckets:
        out.append(_dxfir_from_prims(b))
    return out


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


def _cluster_labels(centers: np.ndarray, *, k: int) -> np.ndarray:
    """Lightweight k-means (no sklearn dependency in training path)."""
    n = centers.shape[0]
    # init: farthest-point / quantile along principal axis (x+y)
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
