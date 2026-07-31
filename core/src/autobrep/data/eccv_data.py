"""ECCV conditioned AutoBrep data: 3 render images + structured TechDraw DXF."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image

from autobrep.data.abc_data import ARDataModule
from autobrep.data.serialize import deserialize_array
from autobrep.data.techdraw_dxf import (
    assign_loop_groups,
    empty_dxf_view_tensors,
    extract_dxf_primitives,
    extract_svg_primitives,
    filter_and_merge,
    merge_dxfir,
    split_into_views,
    tensorize_dxf_views,
    tensors_to_torch,
)
from autobrep.data.techdraw_dxf.schema import DxfIR


VIEW_SIZE = 224
RENDER_COLS = (
    "render_transparent",
    "render_hlg",
    "render_hlg_translucent",
)


def _load_rgb_png(path: Path, size: int = VIEW_SIZE) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    return (np.asarray(img).astype(np.float32) / 255.0).transpose(2, 0, 1)


def load_render_views(
    dataset_root: Path,
    row: Dict[str, Any],
    size: int = VIEW_SIZE,
) -> np.ndarray:
    """Load 3 render PNGs → (3, 3, H, W) float32 in [0, 1]."""
    root = Path(dataset_root)
    views: list[np.ndarray] = []
    for col in RENDER_COLS:
        rel = row.get(col, "")
        path = root / rel if rel else None
        if path is not None and path.is_file():
            views.append(_load_rgb_png(path, size=size))
        else:
            views.append(np.ones((3, size, size), dtype=np.float32))
    return np.stack(views, axis=0).astype(np.float32)


def load_techdraw_geometry(
    dataset_root: Path, row: Dict[str, Any]
) -> dict[str, torch.Tensor]:
    """
    Geometric TechDraw → tensors (V, N, ...).

    Canonical pipeline (locked):
      DXF (+ compatible SVG) → filter_and_merge → merge_dxfir
      → split_into_views: XY-Cut view *regions* then bbox-overlap assign
      → assign_loop_groups per view → tensorize (local bbox normalize)

    Do **not** cluster primitives with k-means as the primary split.
    """
    root = Path(dataset_root)
    parts = []
    dxf_rel = row.get("techdraw_dxf_path", "")
    svg_rel = row.get("techdraw_svg_path", "")
    dxf_path = root / dxf_rel if dxf_rel else None
    svg_path = root / svg_rel if svg_rel else None
    try:
        if dxf_path is not None and dxf_path.is_file():
            parts.append(extract_dxf_primitives(dxf_path))
    except Exception:  # noqa: BLE001
        pass
    try:
        if svg_path is not None and svg_path.is_file():
            parts.append(extract_svg_primitives(svg_path))
    except Exception:  # noqa: BLE001
        pass
    if not parts:
        return empty_dxf_view_tensors()
    try:
        merged = filter_and_merge(merge_dxfir(parts))
        views = split_into_views(merged)
        # Per-view loop grouping (group_id / group_role); missing → defaults 0.
        grouped: list[DxfIR] = []
        for v in views:
            prims = assign_loop_groups(list(v.prims[: int(v.n_prims)]))
            grouped.append(
                DxfIR(
                    n_prims=len(prims),
                    prims=prims,
                    bbox_min=v.bbox_min,
                    bbox_max=v.bbox_max,
                )
            )
        return tensors_to_torch(tensorize_dxf_views(grouped))
    except Exception:  # noqa: BLE001
        return empty_dxf_view_tensors()


# backward-compatible alias
def load_techdraw_dxf(dataset_root: Path, row: Dict[str, Any]) -> dict[str, torch.Tensor]:
    return load_techdraw_geometry(dataset_root, row)


class ECCVViewDataModule(ARDataModule):
    """
    ARDataModule + 3 render images + TechDraw geometry (DXF+SVG, 3 sheet views).

    TechDraw is never rasterized; primitives are set-encoded per orthographic view.
    """

    def __init__(
        self,
        dataset_root: Optional[str] = None,
        view_size: int = VIEW_SIZE,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._dataset_root = (
            Path(dataset_root).resolve()
            if dataset_root
            else Path(self.hparams.data_root).resolve().parents[1]
        )
        self._view_size = int(view_size)

    @property
    def columns(self) -> List[str]:
        cols = super().columns
        extra = [
            "stem",
            "sample_id",
            "num_faces",
            "num_edges",
            "render_transparent",
            "render_hlg",
            "render_hlg_translucent",
            "techdraw_svg_path",
            "techdraw_dxf_path",
        ]
        for c in extra:
            if c not in cols:
                cols = cols + [c]
        return cols

    def _filter_expr(self):
        import pyarrow.compute as pc

        return (pc.field("num_faces") >= self.hparams.min_face) & (
            pc.field("num_faces") <= self.hparams.max_face
        )

    def _dataset_columns(self) -> List[str]:
        return list(dict.fromkeys(self.columns))

    def pre_filter(self, row: Dict[str, Any]) -> bool:
        TOL = 1 / (2 ** (self.hparams.bit - 1))

        face_edge_adj = deserialize_array(row["face_edge_incidence"])
        if len(face_edge_adj) == 0:
            return False
        if len(face_edge_adj.shape) == ():
            return False
        if np.any(np.all(np.logical_not(face_edge_adj), axis=1)):
            return False
        if np.any(np.sum(face_edge_adj.sum(0) != 2)):
            return False
        if face_edge_adj.shape[0] > self.hparams.max_face:
            return False
        if face_edge_adj.shape[1] > self.hparams.max_edge:
            return False

        face_pos = deserialize_array(row["face_bbox_world"])
        xyz_diff = np.abs(face_pos[:, 0:3] - face_pos[:, 3:6])
        if np.any(np.all(xyz_diff < TOL, axis=-1)):
            return False

        edge_pos = deserialize_array(row["edge_bbox_world"])
        xyz_diff = np.abs(edge_pos[:, 0:3] - edge_pos[:, 3:6])
        if np.any(np.all(xyz_diff < TOL, axis=-1)):
            return False

        return True

    def unpickle(self, row: Dict[str, Any]) -> Dict[str, Any]:
        data = super().unpickle(row)
        for key in (
            "stem",
            "sample_id",
            "render_transparent",
            "render_hlg",
            "render_hlg_translucent",
            "techdraw_svg_path",
            "techdraw_dxf_path",
        ):
            if key in row:
                data[key] = row[key]
        return data

    def map_func(self, row: Dict[str, Any], aug: bool) -> Dict[str, Any]:
        output = super().map_func(row, aug=aug)
        output["images"] = load_render_views(
            self._dataset_root, row, size=self._view_size
        )
        dxf = load_techdraw_geometry(self._dataset_root, row)
        output.update(dxf)
        sid = row.get("sample_id") or row.get("stem") or ""
        output["sample_id"] = str(sid)
        # GT cardinality for topology aux head (0 when unavailable).
        output["num_faces"] = int(row.get("num_faces") or 0)
        output["num_edges"] = int(row.get("num_edges") or 0)
        return output

    @property
    def dtypes(self):
        output = super().dtypes
        output["images"] = torch.float32
        output["prim_types"] = torch.int64
        output["prim_linetypes"] = torch.int64
        output["prim_geom"] = torch.float32
        output["prim_mask"] = torch.bool
        output["prim_group_ids"] = torch.int64
        output["prim_group_roles"] = torch.int64
        output["num_faces"] = torch.int64
        output["num_edges"] = torch.int64
        return output
