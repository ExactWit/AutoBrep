#!/usr/bin/env python3
"""exp_launcher infer entry: unconditional AutoBrep sampling → product-domain STEP."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from pytorch_lightning import seed_everything

from autobrep.inference.brepgen_brep_builder import AutoBrepBuilder
from autobrep.inference.inference_common import (
    reconstruct_compound,
    save_debug_images,
    save_point_grid,
)
from autobrep.models.autoregressive import ARGenCheckpointPaths, AutoRegressiveSampler
from autobrep.utils import DotDict, generate_random_string, timer
from occwl.io import save_step as save_step_func


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def build_sample_config(args: argparse.Namespace) -> DotDict:
    return DotDict(
        {
            "api_version": {"major_version": 1, "minor_version": 0, "build_number": 0},
            "seed": {
                "use_seed": bool(args.use_seed),
                "seed_value": int(args.seed),
            },
            "debug_mode": bool(args.debug),
            "output_format": "step",
            "hyper_parameters": {
                "num_batches_to_sample": int(args.num_batches),
                "num_samples_to_generate": int(args.num_batches) * int(args.batch_size),
                "batch_size": int(args.batch_size),
                "complexity": str(args.complexity),
                "temperature": float(args.temperature),
                "sample_method": {
                    "name": "top_p",
                    "top_p_threshold": float(args.top_p),
                },
                "vertex_threshold": float(args.vertex_threshold),
                "sewing_tolerance": float(args.sewing_tolerance),
                "z_threshold": float(args.z_threshold),
            },
            "weight_folder": str(args.weight_folder),
        }
    )


def resolve_weight_folder(args: argparse.Namespace) -> Path:
    if args.weight_folder:
        folder = Path(args.weight_folder)
        if folder.is_dir():
            return folder
        raise FileNotFoundError(f"weight folder not found: {folder}")

    if args.checkpoint:
        ckpt = Path(args.checkpoint)
        if ckpt.is_file():
            return ckpt.parent
        if ckpt.is_dir():
            return ckpt

    if args.exp_dir:
        ckpt_dir = Path(args.exp_dir) / "checkpoints"
        if (ckpt_dir / "ar.ckpt").is_file():
            return ckpt_dir

    raise FileNotFoundError(
        "need --weight-folder, or --checkpoint / --exp-dir with ar.ckpt + FSQ ckpts"
    )


def sample_batch(
    config: DotDict,
    model: AutoRegressiveSampler,
    *,
    infer_dir: Path,
    debug_dir: Path | None,
    point_cloud: Any = None,
) -> dict[str, Any]:
    stem = generate_random_string(20)
    builders = [
        AutoBrepBuilder(
            device=model.device,
            z_threshold=config.hyper_parameters.z_threshold,
            vertex_threshold=config.hyper_parameters.vertex_threshold,
            sewing_tolerance=config.hyper_parameters.sewing_tolerance,
        )
    ]

    log: dict[str, Any] = {
        "stem": stem,
        "valid": 0,
        "invalid": 0,
        "files": [],
        "errors": [],
    }

    samples = model.sample_tokens(
        config=config,
        batch_size=config.hyper_parameters.batch_size,
        point_cloud=point_cloud,
    )
    batch_decoded = model.decode_tokens(samples.detach().cpu().numpy())
    batch_cad_data = model.convert_to_cad_data(batch_decoded)

    with timer("Time to rebuild: %s seconds"):
        for sample_idx, cad_data in enumerate(batch_cad_data):
            sample_stem = f"{stem}_{str(sample_idx).zfill(3)}"
            try:
                if config.debug_mode and debug_dir is not None:
                    save_point_grid(
                        debug_dir / f"{sample_stem}_before_joint_optimize.npz",
                        cad_data,
                    )
                    save_debug_images(
                        cad_data,
                        debug_dir / f"{sample_stem}_face.png",
                        debug_dir / f"{sample_stem}_edge.png",
                    )

                result = reconstruct_compound(cad_data, builders)
                if result is None:
                    log["invalid"] += 1
                    log["errors"].append({"sample": sample_stem, "error": "rebuild_failed"})
                    continue

                step_path = infer_dir / f"{sample_stem}.step"
                save_step_func([result], step_path)
                sidecar = {
                    "sample_id": sample_stem,
                    "complexity": config.hyper_parameters.complexity,
                    "temperature": config.hyper_parameters.temperature,
                    "top_p": config.hyper_parameters.sample_method.top_p_threshold,
                    "pc_conditioned": bool(getattr(model, "pc_conditioned", False)),
                    "files": [f"infer/{sample_stem}.step"],
                }
                (infer_dir / f"{sample_stem}.json").write_text(
                    json.dumps(sidecar, indent=2), encoding="utf-8"
                )
                log["valid"] += 1
                log["files"].append(f"infer/{sample_stem}.step")
            except Exception as exc:  # noqa: BLE001 — keep batch going
                log["invalid"] += 1
                log["errors"].append({"sample": sample_stem, "error": str(exc)})

    return log


def load_point_cloud_npy(path: str, batch_size: int) -> torch.Tensor:
    import numpy as np

    pts = np.load(path)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"point cloud must be (N,3), got {pts.shape}")
    # normalize like training
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    center = 0.5 * (mins + maxs)
    pts = pts - center
    scale = np.abs(pts).max()
    if scale < 1e-8:
        scale = 1.0
    pts = (pts / scale).astype("float32")
    tensor = torch.from_numpy(pts).unsqueeze(0).repeat(batch_size, 1, 1)
    return tensor


def write_manifest(
    output_dir: Path,
    *,
    dataset: str,
    task: str,
    config: DotDict,
    batch_logs: list[dict[str, Any]],
) -> None:
    samples = []
    for batch in batch_logs:
        for rel in batch.get("files") or []:
            sample_id = Path(rel).stem
            samples.append(
                {
                    "sample_id": sample_id,
                    "files": [rel, f"infer/{sample_id}.json"],
                }
            )
    payload = {
        "dataset": dataset,
        "task": task,
        "repo": "AutoBrep",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": {
            "weight_folder": config.weight_folder,
            "complexity": config.hyper_parameters.complexity,
            "batch_size": config.hyper_parameters.batch_size,
            "num_batches": config.hyper_parameters.num_batches_to_sample,
            "temperature": config.hyper_parameters.temperature,
            "top_p": config.hyper_parameters.sample_method.top_p_threshold,
            "vertex_threshold": config.hyper_parameters.vertex_threshold,
            "sewing_tolerance": config.hyper_parameters.sewing_tolerance,
            "z_threshold": config.hyper_parameters.z_threshold,
            "seed": config.seed,
            "debug_mode": config.debug_mode,
        },
        "summary": {
            "valid": sum(b.get("valid", 0) for b in batch_logs),
            "invalid": sum(b.get("invalid", 0) for b in batch_logs),
        },
        "samples": samples,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AutoBrep infer for exp_launcher")
    p.add_argument("--exp-dir", default="")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--data-dir", default="")
    p.add_argument("--datasplit", default="")
    p.add_argument("--dataset", default="abc")
    p.add_argument("--task", default="gen")
    p.add_argument("--checkpoint", default="")
    p.add_argument("--weight-folder", default="")
    p.add_argument("--gpu", default="0")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-batches", type=int, default=10)
    p.add_argument(
        "--complexity",
        default="medium",
        choices=["random", "easy", "medium", "hard"],
    )
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--vertex-threshold", type=float, default=0.002)
    p.add_argument("--sewing-tolerance", type=float, default=0.002)
    p.add_argument("--z-threshold", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=689447)
    p.add_argument("--use-seed", type=_parse_bool, default=False)
    p.add_argument("--debug", type=_parse_bool, default=True)
    p.add_argument("--format", default="step")
    p.add_argument("--index", default="")
    p.add_argument("--sample-id", default="")
    p.add_argument("--resume-from", default="")
    p.add_argument("--pc-conditioned", type=_parse_bool, default=False)
    p.add_argument("--point-cloud", default="", help="Optional .npy (N,3) for PC infer")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    weight_folder = resolve_weight_folder(args)
    args.weight_folder = str(weight_folder)

    output_dir = Path(args.output_dir)
    infer_dir = output_dir / "infer"
    infer_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = infer_dir / "debug" if args.debug else None
    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)

    if args.gpu != "":
        import os

        os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))

    config = build_sample_config(args)
    checkpoints = ARGenCheckpointPaths.from_folder(folder=str(weight_folder))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pc_ckpt = None
    if args.pc_conditioned:
        if args.checkpoint and Path(args.checkpoint).is_file():
            pc_ckpt = args.checkpoint
        elif (Path(args.exp_dir) / "checkpoints" / "last.ckpt").is_file():
            pc_ckpt = str(Path(args.exp_dir) / "checkpoints" / "last.ckpt")
        else:
            pc_ckpt = str(weight_folder / "last.ckpt") if False else None
            # fall back: try last.ckpt under weight folder
            cand = Path(weight_folder) / "last.ckpt"
            if cand.is_file():
                pc_ckpt = str(cand)
        if not pc_ckpt:
            raise FileNotFoundError(
                "pc-conditioned infer needs --checkpoint (PC Lightning ckpt) "
                "or {exp-dir}/checkpoints/last.ckpt"
            )

    model = AutoRegressiveSampler(
        checkpoint_paths=checkpoints,
        device=device,
        pc_conditioned=bool(args.pc_conditioned),
        pc_ckpt=pc_ckpt,
    )

    point_cloud = None
    if args.pc_conditioned:
        if not args.point_cloud:
            raise ValueError("--pc-conditioned requires --point-cloud /path/to.npy")
        point_cloud = load_point_cloud_npy(args.point_cloud, args.batch_size)

    if config.seed.use_seed:
        seed_everything(config.seed.seed_value, workers=True)

    print(
        f"[AutoBrep infer] weight_folder={weight_folder} "
        f"pc_conditioned={args.pc_conditioned} "
        f"batches={args.num_batches} batch_size={args.batch_size} "
        f"complexity={args.complexity} → {infer_dir}",
        flush=True,
    )

    batch_logs: list[dict[str, Any]] = []
    t0 = time.time()
    for i in range(int(args.num_batches)):
        print(f"------------ batch {i} ------------", flush=True)
        batch_logs.append(
            sample_batch(
                config,
                model,
                infer_dir=infer_dir,
                debug_dir=debug_dir,
                point_cloud=point_cloud,
            )
        )
    write_manifest(
        output_dir,
        dataset=args.dataset,
        task=args.task,
        config=config,
        batch_logs=batch_logs,
    )
    print(f"[AutoBrep infer] done in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
