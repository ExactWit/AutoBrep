"""Convert DxfIR <-> training tensors."""

from __future__ import annotations

import numpy as np
import torch

from autobrep.data.techdraw_dxf.schema import (
    GEOM_DIM,
    LINETYPE_CENTER,
    LINETYPE_HIDDEN,
    LINETYPE_OTHER,
    LINETYPE_SOLID,
    MAX_PRIMS,
    SPLINE_SAMPLE_POINTS,
    DxfIR,
    PrimIR,
)

_LINETYPE_TO_ID = {
    "solid": LINETYPE_SOLID,
    "hidden": LINETYPE_HIDDEN,
    "center": LINETYPE_CENTER,
    "other": LINETYPE_OTHER,
}


def _norm_xy(xy, bbox_min: np.ndarray, span: np.ndarray) -> np.ndarray:
    return ((np.asarray(xy, dtype=np.float32)[:2] - bbox_min) / span).astype(np.float32)


def _norm_scalar(value: float, scale: float) -> float:
    return float(value) / max(float(scale), 1e-6)


def _geom_vec(prim: PrimIR, bbox_min: np.ndarray, span: np.ndarray, scale: float) -> np.ndarray:
    out = np.zeros((GEOM_DIM,), dtype=np.float32)
    params = prim.params
    if prim.type == "line":
        out[0:2] = _norm_xy(params["start"], bbox_min, span)
        out[2:4] = _norm_xy(params["end"], bbox_min, span)
    elif prim.type in {"arc", "circle"}:
        out[0:2] = _norm_xy(params["center"], bbox_min, span)
        out[2] = _norm_scalar(float(params["radius"]), scale)
        out[3] = float(params.get("start_angle", 0.0))
        out[4] = float(params.get("end_angle", 2.0 * np.pi))
    elif prim.type == "ellipse":
        out[0:2] = _norm_xy(params["center"], bbox_min, span)
        maj = np.asarray(params.get("major_axis") or [1, 0], dtype=np.float32)[:2]
        out[2:4] = maj / max(scale, 1e-6)
        out[4] = float(params.get("ratio", 1.0))
        out[5] = float(params.get("start_param", 0.0))
        out[6] = float(params.get("end_param", 2.0 * np.pi))
    elif prim.type == "spline":
        cps = list(params.get("control_points") or [])
        if len(cps) == 0:
            return out
        idxs = np.linspace(0, len(cps) - 1, num=SPLINE_SAMPLE_POINTS).astype(int)
        for i, idx in enumerate(idxs):
            out[2 * i : 2 * i + 2] = _norm_xy(cps[idx], bbox_min, span)
    elif prim.type == "lwpolyline":
        pts = list(params.get("points") or [])
        if not pts:
            return out
        out[0:2] = _norm_xy(pts[0], bbox_min, span)
        out[2:4] = _norm_xy(pts[-1], bbox_min, span)
        mid = pts[len(pts) // 2]
        out[4:6] = _norm_xy(mid, bbox_min, span)
        out[6] = 1.0 if params.get("closed") else 0.0
    elif prim.type == "other" and "center" in params:
        out[0:2] = _norm_xy(params["center"], bbox_min, span)
    return out


def tensorize_dxf(dxfir: DxfIR, max_prims: int = MAX_PRIMS) -> dict[str, np.ndarray]:
    n_prims = min(int(dxfir.n_prims), max_prims)
    prim_types = np.zeros((max_prims,), dtype=np.int64)
    prim_linetypes = np.zeros((max_prims,), dtype=np.int64)
    prim_geom = np.zeros((max_prims, GEOM_DIM), dtype=np.float32)
    prim_mask = np.zeros((max_prims,), dtype=np.bool_)

    bbox_min = np.asarray(dxfir.bbox_min, dtype=np.float32)[:2]
    bbox_max = np.asarray(dxfir.bbox_max, dtype=np.float32)[:2]
    span = np.maximum(bbox_max - bbox_min, 1e-6).astype(np.float32)
    scale = float(np.max(span))

    for idx, prim in enumerate(dxfir.prims[:n_prims]):
        prim_types[idx] = prim.type_id
        prim_linetypes[idx] = _LINETYPE_TO_ID.get(prim.linetype, LINETYPE_OTHER)
        prim_geom[idx] = _geom_vec(prim, bbox_min, span, scale)
        prim_mask[idx] = True

    return {
        "n_prims": np.asarray(n_prims, dtype=np.int64),
        "prim_types": prim_types,
        "prim_linetypes": prim_linetypes,
        "prim_geom": prim_geom,
        "prim_mask": prim_mask,
        "dxf_bbox_min": bbox_min.astype(np.float32),
        "dxf_bbox_max": bbox_max.astype(np.float32),
    }


def empty_dxf_tensors(max_prims: int = MAX_PRIMS) -> dict[str, torch.Tensor]:
    return {
        "n_prims": torch.tensor(0, dtype=torch.long),
        "prim_types": torch.zeros((max_prims,), dtype=torch.long),
        "prim_linetypes": torch.zeros((max_prims,), dtype=torch.long),
        "prim_geom": torch.zeros((max_prims, GEOM_DIM), dtype=torch.float32),
        "prim_mask": torch.zeros((max_prims,), dtype=torch.bool),
        "dxf_bbox_min": torch.zeros(2, dtype=torch.float32),
        "dxf_bbox_max": torch.ones(2, dtype=torch.float32),
    }


def tensors_to_torch(tensors: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for key, value in tensors.items():
        if key == "n_prims":
            out[key] = torch.tensor(int(value), dtype=torch.long)
        elif key in {"prim_types", "prim_linetypes"}:
            out[key] = torch.from_numpy(np.asarray(value)).long()
        elif key == "prim_mask":
            out[key] = torch.from_numpy(np.asarray(value)).bool()
        else:
            out[key] = torch.from_numpy(np.asarray(value)).float()
    return out
