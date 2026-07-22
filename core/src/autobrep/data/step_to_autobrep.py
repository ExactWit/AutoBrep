"""STEP → AutoBrep-compatible UV grids / bbox / face-edge incidence.

Sampling follows AutoBrep / BRepGen conventions:
  - Face: 32×32 points uniformly in the face's UV parameter domain
    (``BRepAdaptor_Surface(face, True)`` = face-restricted bounds).
  - Edge: 32 points uniformly in the curve parameter domain.
  - Solid mapped into ≈[-1, 1] via AABB center + half max-extent (ABC style).
  - Per-primitive NCS relative to each face/edge bbox.
  - UV-origin invariance applied at write time via ``sort_uv_grids`` /
    ``sort_u_grids`` (same as training ``uv_invariant=True``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_FACE
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopTools import TopTools_IndexedMapOfShape
from OCC.Core.TopoDS import topods
from OCC.Extend.TopologyUtils import TopologyExplorer

from autobrep.utils import compute_bbox_center_and_size, sort_u_grids, sort_uv_grids

FACE_UV = 32
EDGE_U = 32


@dataclass
class StepExtractResult:
    face_points_normalized: np.ndarray  # (F, 32, 32, 3)
    edge_points_normalized: np.ndarray  # (E, 32, 3)
    face_bbox_world: np.ndarray  # (F, 6) solid-normalized
    edge_bbox_world: np.ndarray  # (E, 6)
    face_edge_incidence: np.ndarray  # (F, E) bool
    num_faces: int
    num_edges: int


class StepExtractError(Exception):
    """Non-recoverable extract failure (skip sample)."""


def read_step_shape(step_path: Path):
    reader = STEPControl_Reader()
    status = reader.ReadFile(str(step_path))
    if status != IFSelect_RetDone:
        raise StepExtractError(f"STEP read failed: {step_path}")
    reader.TransferRoots()
    shape = reader.OneShape()
    if shape is None or shape.IsNull():
        raise StepExtractError(f"Empty STEP shape: {step_path}")
    return shape


def _sample_face_grid(face, n: int = FACE_UV) -> np.ndarray:
    """Uniform UV grid on the face-restricted parameter domain (AutoBrep)."""
    adaptor = BRepAdaptor_Surface(face, True)
    u_min, u_max = adaptor.FirstUParameter(), adaptor.LastUParameter()
    v_min, v_max = adaptor.FirstVParameter(), adaptor.LastVParameter()
    if abs(u_max - u_min) < 1e-12:
        u_max = u_min + 1.0
    if abs(v_max - v_min) < 1e-12:
        v_max = v_min + 1.0
    # Equally spaced including endpoints — matches AutoBrep N×N grid.
    us = np.linspace(u_min, u_max, n, dtype=np.float64)
    vs = np.linspace(v_min, v_max, n, dtype=np.float64)
    pts = np.zeros((n, n, 3), dtype=np.float64)
    for i, u in enumerate(us):
        for j, v in enumerate(vs):
            p = adaptor.Value(float(u), float(v))
            pts[i, j] = (p.X(), p.Y(), p.Z())
    return pts


def _sample_edge_curve(edge, n: int = EDGE_U) -> np.ndarray:
    adaptor = BRepAdaptor_Curve(edge)
    t0, t1 = adaptor.FirstParameter(), adaptor.LastParameter()
    if abs(t1 - t0) < 1e-12:
        t1 = t0 + 1.0
    ts = np.linspace(t0, t1, n, dtype=np.float64)
    pts = np.zeros((n, 3), dtype=np.float64)
    for i, t in enumerate(ts):
        p = adaptor.Value(float(t))
        pts[i] = (p.X(), p.Y(), p.Z())
    return pts


def _bbox6(points: np.ndarray) -> np.ndarray:
    flat = points.reshape(-1, 3)
    mn = flat.min(axis=0)
    mx = flat.max(axis=0)
    return np.concatenate([mn, mx]).astype(np.float64)


def _to_ncs(points: np.ndarray, bbox6: np.ndarray) -> np.ndarray:
    center, size = compute_bbox_center_and_size(bbox6[:3], bbox6[3:])
    if size < 1e-12:
        size = 1.0
    return ((points - center) / (size * 0.5)).astype(np.float32)


def _solid_normalize_inplace(
    face_wcs: list[np.ndarray], edge_wcs: list[np.ndarray]
) -> None:
    """Map samples into solid NCS ≈ [-1, 1] (ABC / AutoBrep parquet convention)."""
    all_pts = []
    for g in face_wcs:
        all_pts.append(g.reshape(-1, 3))
    for e in edge_wcs:
        all_pts.append(e.reshape(-1, 3))
    if not all_pts:
        raise StepExtractError("No geometry to normalize")
    cloud = np.concatenate(all_pts, axis=0)
    mn, mx = cloud.min(axis=0), cloud.max(axis=0)
    center = 0.5 * (mn + mx)
    extent = float(np.max(mx - mn))
    scale = extent * 0.5 if extent > 1e-12 else 1.0
    for i in range(len(face_wcs)):
        face_wcs[i] = (face_wcs[i] - center) / scale
    for i in range(len(edge_wcs)):
        edge_wcs[i] = (edge_wcs[i] - center) / scale


def extract_autobrep_from_shape(
    shape,
    max_face: int = 200,
    max_edge: int = 1000,
    *,
    apply_uv_sort: bool = True,
) -> StepExtractResult:
    """Extract AutoBrep arrays from an OCC shape. Raises StepExtractError on skip."""
    face_map = TopTools_IndexedMapOfShape()
    edge_map = TopTools_IndexedMapOfShape()

    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        face_map.Add(exp.Current())
        exp.Next()
    exp = TopExp_Explorer(shape, TopAbs_EDGE)
    while exp.More():
        edge_map.Add(exp.Current())
        exp.Next()

    num_faces = face_map.Size()
    num_edges = edge_map.Size()
    if num_faces < 2:
        raise StepExtractError(f"too few faces: {num_faces}")
    if num_faces > max_face:
        raise StepExtractError(f"too many faces: {num_faces} > {max_face}")
    if num_edges > max_edge:
        raise StepExtractError(f"too many edges: {num_edges} > {max_edge}")
    if num_edges < 1:
        raise StepExtractError("no edges")

    face_wcs: list[np.ndarray] = []
    for i in range(1, num_faces + 1):
        face = topods.Face(face_map.FindKey(i))
        face_wcs.append(_sample_face_grid(face))

    edge_wcs: list[np.ndarray] = []
    for i in range(1, num_edges + 1):
        edge = topods.Edge(edge_map.FindKey(i))
        edge_wcs.append(_sample_edge_curve(edge))

    _solid_normalize_inplace(face_wcs, edge_wcs)

    face_bbox = np.stack([_bbox6(g) for g in face_wcs], axis=0).astype(np.float32)
    edge_bbox = np.stack([_bbox6(e) for e in edge_wcs], axis=0).astype(np.float32)
    face_ncs = np.stack(
        [_to_ncs(g, face_bbox[i]) for i, g in enumerate(face_wcs)], axis=0
    ).astype(np.float32)
    edge_ncs = np.stack(
        [_to_ncs(e, edge_bbox[i]) for i, e in enumerate(edge_wcs)], axis=0
    ).astype(np.float32)

    # AutoBrep UV-origin convention (lexicographically lowest corner at u_min,v_min).
    if apply_uv_sort:
        face_ncs = sort_uv_grids(face_ncs).astype(np.float32)
        edge_ncs = sort_u_grids(edge_ncs).astype(np.float32)

    incidence = np.zeros((num_faces, num_edges), dtype=bool)
    for fi in range(1, num_faces + 1):
        face = topods.Face(face_map.FindKey(fi))
        topo = TopologyExplorer(face)
        for edge in topo.edges():
            ei = edge_map.FindIndex(edge)
            if ei > 0:
                incidence[fi - 1, ei - 1] = True

    edge_face_count = incidence.sum(axis=0)
    if np.any(edge_face_count != 2):
        raise StepExtractError(
            f"non-manifold edges: counts={np.unique(edge_face_count, return_counts=True)}"
        )
    if np.any(~incidence.any(axis=1)):
        raise StepExtractError("face with no edges")

    return StepExtractResult(
        face_points_normalized=face_ncs,
        edge_points_normalized=edge_ncs,
        face_bbox_world=face_bbox,
        edge_bbox_world=edge_bbox,
        face_edge_incidence=incidence,
        num_faces=num_faces,
        num_edges=num_edges,
    )


def extract_autobrep_from_step(
    step_path: Path | str,
    max_face: int = 200,
    max_edge: int = 1000,
    *,
    apply_uv_sort: bool = True,
) -> StepExtractResult:
    shape = read_step_shape(Path(step_path))
    return extract_autobrep_from_shape(
        shape, max_face=max_face, max_edge=max_edge, apply_uv_sort=apply_uv_sort
    )


RENDER_VIEW_SUBDIRS = (
    ("render_transparent", "render_3d/transparent_shaded_edges_perspective"),
    ("render_hlg", "render_3d/hlg_perspective"),
    ("render_hlg_translucent", "render_3d/hlg_translucent_faces_perspective"),
)

# backward-compatible alias (train-only absolute relpaths)
RENDER_VIEWS = tuple(
    (col, f"train/{sub}") for col, sub in RENDER_VIEW_SUBDIRS
)


def condition_root_for_split(split: str | None) -> str:
    """Datasplit name → on-disk condition root (train/ vs test_public/)."""
    name = str(split or "train").strip().lower()
    if name in {"public_test", "public", "test_public"}:
        return "test_public"
    return "train"


def eccv_condition_paths(
    sample_id: str,
    *,
    condition_root: str = "train",
    split: str | None = None,
) -> dict[str, str]:
    """Relative paths (posix) for parquet / infer condition columns."""
    sid = sample_id.strip()
    root = condition_root_for_split(split) if split is not None else str(condition_root)
    out: dict[str, str] = {}
    for col, sub in RENDER_VIEW_SUBDIRS:
        out[col] = f"{root}/{sub}/{sid}.png"
    out["techdraw_svg_path"] = f"{root}/techdraw/svg/{sid}.svg"
    out["techdraw_dxf_path"] = f"{root}/techdraw/dxf/{sid}.dxf"
    return out


def result_to_row(
    sample_id: str,
    result: StepExtractResult,
    *,
    include_condition_paths: bool = True,
) -> dict[str, Any]:
    from autobrep.data.serialize import serialize_array

    row: dict[str, Any] = {
        "stem": sample_id,
        "sample_id": sample_id,
        "num_faces": int(result.num_faces),
        "num_edges": int(result.num_edges),
        "face_points_normalized": serialize_array(result.face_points_normalized),
        "edge_points_normalized": serialize_array(result.edge_points_normalized),
        "face_bbox_world": serialize_array(result.face_bbox_world),
        "edge_bbox_world": serialize_array(result.edge_bbox_world),
        "face_edge_incidence": serialize_array(result.face_edge_incidence),
    }
    if include_condition_paths:
        row.update(eccv_condition_paths(sample_id))
    return row
