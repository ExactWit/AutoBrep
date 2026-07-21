"""Point-cloud condition helpers for PC-conditioned AutoBrep infer."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from autobrep.data.serialize import deserialize_array
from autobrep.utils import ncs2wcs


def normalize_points_np(pts: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Center + scale to fit [-1, 1] (same as training sample_point_cloud)."""
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 3)
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    center = 0.5 * (mins + maxs)
    pts = pts - center
    scale = float(np.abs(pts).max())
    if scale < eps:
        scale = 1.0
    return (pts / scale).astype(np.float32)


def sample_point_cloud_from_faces(
    face_pos: np.ndarray,
    face_ncs: np.ndarray,
    *,
    num_points: int = 2048,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Match ARDataModule.sample_point_cloud (UV grids → WCS → subsample → normalize)."""
    rng = rng or np.random.default_rng(0)
    face_pos = np.asarray(face_pos).reshape(-1, 6)
    face_ncs = np.asarray(face_ncs)
    wcs_list = []
    for bbox, ncs in zip(face_pos, face_ncs):
        if not np.any(ncs):
            continue
        wcs_list.append(ncs2wcs(ncs, bbox).reshape(-1, 3))
    if not wcs_list:
        return np.zeros((num_points, 3), dtype=np.float32)
    pts = np.concatenate(wcs_list, axis=0).astype(np.float32)
    if pts.shape[0] >= num_points:
        idx = rng.choice(pts.shape[0], num_points, replace=False)
    else:
        idx = rng.choice(pts.shape[0], num_points, replace=True)
    return normalize_points_np(pts[idx])


def load_point_cloud_npy(
    path: Union[str, Path],
    *,
    batch_size: int = 1,
    num_points: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Load (N,3) .npy, optionally resample to num_points, normalize, tile to (B,N,3).
    """
    rng = rng or np.random.default_rng(0)
    path = Path(path)
    pts = np.load(path)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"point cloud must be (N,3), got {pts.shape}")
    pts = pts.astype(np.float32)
    meta: Dict[str, Any] = {
        "source": "npy",
        "path": str(path),
        "raw_num_points": int(pts.shape[0]),
    }
    if num_points is not None and pts.shape[0] != num_points:
        if pts.shape[0] >= num_points:
            idx = rng.choice(pts.shape[0], num_points, replace=False)
        else:
            idx = rng.choice(pts.shape[0], num_points, replace=True)
        pts = pts[idx]
    pts = normalize_points_np(pts)
    meta["num_points"] = int(pts.shape[0])
    tensor = torch.from_numpy(pts).unsqueeze(0).repeat(batch_size, 1, 1)
    return tensor, meta


def find_abc_row_by_stem(
    data_root: Union[str, Path],
    stem: str,
    *,
    split: str = "train",
    max_files: int = 0,
) -> Dict[str, Any]:
    """Scan parquet shards for an exact stem match."""
    import pyarrow.parquet as pq

    root = Path(data_root)
    split_dir = root / split
    if not split_dir.is_dir() and (root.parent / split).is_dir():
        split_dir = root.parent / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"missing split dir: {root / split}")

    cols = [
        "stem",
        "face_points_normalized",
        "face_bbox_world",
        "num_faces",
    ]
    files = sorted(split_dir.glob("*.parquet"))
    if max_files and max_files > 0:
        files = files[:max_files]
    if not files:
        raise FileNotFoundError(f"no parquet under {split_dir}")

    for f in files:
        try:
            stems = pq.read_table(f, columns=["stem"]).column(0).to_pylist()
            if stem not in stems:
                continue
            idx = stems.index(stem)
            schema = set(pq.read_schema(f).names)
            use_cols = [c for c in cols if c in schema]
            pdf = pq.read_table(f, columns=use_cols).slice(idx, 1).to_pandas()
            return {c: pdf.iloc[0][c] for c in pdf.columns}
        except Exception:
            continue
    raise ValueError(f"stem not found under {split_dir}: {stem}")


def load_point_cloud_from_abc(
    data_root: Union[str, Path],
    stem: str,
    *,
    batch_size: int = 1,
    num_points: int = 2048,
    split: str = "train",
    seed: int = 0,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    row = find_abc_row_by_stem(data_root, stem, split=split)
    face_pos = deserialize_array(row["face_bbox_world"])
    face_ncs = deserialize_array(row["face_points_normalized"])
    rng = np.random.default_rng(seed)
    pts = sample_point_cloud_from_faces(
        face_pos, face_ncs, num_points=num_points, rng=rng
    )
    meta = {
        "source": "abc",
        "stem": stem,
        "split": split,
        "num_faces": int(face_pos.shape[0]),
        "num_points": int(pts.shape[0]),
        "data_root": str(data_root),
    }
    tensor = torch.from_numpy(pts).unsqueeze(0).repeat(batch_size, 1, 1)
    return tensor, meta


def _val_loss_from_name(path: Path) -> float:
    m = re.search(r"val_loss=([0-9.]+)", path.name)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return float("inf")


def _ckpt_looks_loadable(path: Path) -> bool:
    """Cheap zip central-directory check (catches truncated last.ckpt)."""
    try:
        import zipfile

        with zipfile.ZipFile(path, "r") as zf:
            return zf.testzip() is None
    except Exception:
        return False


def resolve_pc_checkpoint(
    *,
    checkpoint: str = "",
    exp_dir: str = "",
    weight_folder: str = "",
) -> Path:
    """
    Resolve a PC Lightning ckpt for infer.

    Preference:
      1) --checkpoint file
      2) --checkpoint dir → best pc-cond-*.ckpt inside (or last.ckpt if valid)
      3) {exp_dir}/checkpoints best / last
      4) {weight_folder}/last.ckpt or pc-cond-*.ckpt
    """
    candidates: List[Path] = []

    if checkpoint:
        p = Path(checkpoint)
        if p.is_file():
            if _ckpt_looks_loadable(p):
                return p
            # Launcher often points at last.ckpt which may be truncated — fall back
            # to sibling best pc-cond-*.ckpt in the same directory.
            print(
                f"[resolve_pc_checkpoint] skip corrupted file: {p} "
                f"(will search siblings)",
                flush=True,
            )
            candidates.append(p.parent)
            # also try parent/parent/checkpoints if somehow nested oddly
            if p.parent.name == "checkpoints" and p.parent.parent.is_dir():
                candidates.append(p.parent)
        elif p.is_dir():
            if (p / "checkpoints").is_dir():
                candidates.append(p / "checkpoints")
            else:
                candidates.append(p)

    if exp_dir:
        ed = Path(exp_dir)
        if (ed / "checkpoints").is_dir():
            candidates.append(ed / "checkpoints")
        elif ed.is_dir():
            candidates.append(ed)

    if weight_folder:
        wf = Path(weight_folder)
        if wf.is_dir():
            candidates.append(wf)

    seen = set()
    for ckpt_dir in candidates:
        key = str(ckpt_dir.resolve()) if ckpt_dir.exists() else str(ckpt_dir)
        if key in seen:
            continue
        seen.add(key)
        if not ckpt_dir.is_dir():
            continue

        scored = sorted(
            [p for p in ckpt_dir.glob("pc-cond*.ckpt") if p.is_file()],
            key=_val_loss_from_name,
        )
        for p in scored:
            if _ckpt_looks_loadable(p):
                return p

        last = ckpt_dir / "last.ckpt"
        if last.is_file() and _ckpt_looks_loadable(last):
            return last

        # any other *.ckpt except ar/fsq
        others = [
            p
            for p in sorted(ckpt_dir.glob("*.ckpt"))
            if p.name not in {"ar.ckpt", "surf-fsq.ckpt", "edge-fsq.ckpt"}
            and _ckpt_looks_loadable(p)
        ]
        if others:
            return others[0]

    raise FileNotFoundError(
        "pc-conditioned infer needs a loadable PC Lightning ckpt via "
        "--checkpoint /path/to.ckpt (or train run dir). "
        "Note: last.ckpt may be truncated; prefer pc-cond-step*-val_loss=*.ckpt"
    )


def save_point_cloud_npy(path: Union[str, Path], points: np.ndarray) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(points, dtype=np.float32))
    return path


def save_point_cloud_preview(
    path: Union[str, Path],
    points: np.ndarray,
    *,
    title: str = "condition point cloud",
    max_points: int = 4096,
) -> Path:
    """
    Save a 2×2 orthographic scatter preview (xy / xz / yz / 3D).
    .npy itself is not viewable; this PNG is the preview artifact.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if pts.shape[0] > max_points:
        rng = np.random.default_rng(0)
        pts = pts[rng.choice(pts.shape[0], max_points, replace=False)]

    # Color by height (y) for quick shape reading
    c = pts[:, 1]
    fig, axes = plt.subplots(2, 2, figsize=(8, 8), dpi=120)
    views = [
        (axes[0, 0], 0, 1, "X", "Y", "XY"),
        (axes[0, 1], 0, 2, "X", "Z", "XZ"),
        (axes[1, 0], 1, 2, "Y", "Z", "YZ"),
    ]
    for ax, i, j, xl, yl, name in views:
        ax.scatter(pts[:, i], pts[:, j], c=c, s=2, cmap="viridis", linewidths=0)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_title(name)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)

    ax3 = fig.add_subplot(2, 2, 4, projection="3d")
    ax3.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=c, s=2, cmap="viridis", linewidths=0)
    ax3.set_xlabel("X")
    ax3.set_ylabel("Y")
    ax3.set_zlabel("Z")
    ax3.set_title("3D")
    try:
        ax3.set_box_aspect((1, 1, 1))
    except Exception:
        pass

    fig.suptitle(f"{title}  (N={pts.shape[0]})", fontsize=11)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path
