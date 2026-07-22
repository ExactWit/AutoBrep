"""Multi-view + TechDraw DXF helpers for ECCV view-conditioned AutoBrep infer."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import torch

from autobrep.data.eccv_data import VIEW_SIZE, load_render_views, load_techdraw_dxf
from autobrep.inference.pc_condition import _ckpt_looks_loadable, _val_loss_from_name


def resolve_view_checkpoint(
    *,
    checkpoint: str = "",
    exp_dir: str = "",
    weight_folder: str = "",
) -> Path:
    candidates: List[Path] = []

    if checkpoint:
        p = Path(checkpoint)
        if p.is_file():
            if _ckpt_looks_loadable(p):
                return p
            print(
                f"[resolve_view_checkpoint] skip corrupted file: {p}",
                flush=True,
            )
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
            [p for p in ckpt_dir.glob("eccv-view*.ckpt") if p.is_file()],
            key=_val_loss_from_name,
        )
        for p in scored:
            if _ckpt_looks_loadable(p):
                return p

        last = ckpt_dir / "last.ckpt"
        if last.is_file() and _ckpt_looks_loadable(last):
            return last

        others = [
            p
            for p in sorted(ckpt_dir.glob("*.ckpt"))
            if p.name not in {"ar.ckpt", "surf-fsq.ckpt", "edge-fsq.ckpt"}
            and _ckpt_looks_loadable(p)
        ]
        if others:
            return others[0]

    raise FileNotFoundError(
        "view-conditioned infer needs a loadable ECCV Lightning ckpt via "
        "--checkpoint /path/to.ckpt (or train run dir). "
        "Prefer eccv-view-*-val_loss=*.ckpt"
    )


def load_condition_for_sample(
    dataset_root: Union[str, Path],
    sample_id: str,
    *,
    batch_size: int = 1,
    size: int = VIEW_SIZE,
    split: str = "val",
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, Any]]:
    """
    Load 3 render views + TechDraw DXF primitives for one sample.

    ``split=public_test`` reads conditions from ``test_public/`` (files differ
    from train/ even when sample ids overlap).

    Returns:
        images: (B, 3, 3, H, W)
        dxf: batched prim_* tensors
        meta: dict
    """
    from autobrep.data.step_to_autobrep import (
        condition_root_for_split,
        eccv_condition_paths,
    )

    root = Path(dataset_root)
    sid = str(sample_id).strip()
    cond_root = condition_root_for_split(split)
    row = eccv_condition_paths(sid, condition_root=cond_root)
    images_np = load_render_views(root, row, size=size)
    images = torch.from_numpy(images_np).unsqueeze(0).expand(batch_size, -1, -1, -1, -1)
    dxf = load_techdraw_dxf(root, row)
    batched: Dict[str, torch.Tensor] = {}
    for k, v in dxf.items():
        if not isinstance(v, torch.Tensor):
            continue
        if v.ndim == 0:
            batched[k] = v.detach().clone().reshape(1).expand(batch_size).contiguous()
        else:
            batched[k] = v.unsqueeze(0).expand(batch_size, *v.shape).contiguous()
    meta: Dict[str, Any] = {
        "source": "eccv_renders_plus_dxf",
        "sample_id": sid,
        "dataset_root": str(root),
        "split": str(split),
        "condition_root": cond_root,
        "paths": row,
        "images_shape": list(images.shape),
        "n_prims": int(dxf["n_prims"].item()) if hasattr(dxf["n_prims"], "item") else int(dxf["n_prims"]),
    }
    return images.contiguous(), batched, meta
