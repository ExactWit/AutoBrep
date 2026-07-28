#!/usr/bin/env python3
"""exp_launcher infer entry: uncond / PC-cond / ECCV view-cond → STEP."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Optional

import torch
from pytorch_lightning import seed_everything

from autobrep.inference.brepgen_brep_builder import AutoBrepBuilder
from autobrep.inference.inference_common import (
    reconstruct_compound,
    save_debug_images,
    save_point_grid,
)
from autobrep.inference.pc_condition import (
    load_point_cloud_from_abc,
    load_point_cloud_npy,
    resolve_pc_checkpoint,
    save_point_cloud_npy,
    save_point_cloud_preview,
)
from autobrep.inference.view_condition import (
    load_condition_for_sample,
    resolve_view_checkpoint,
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
                "postprocess_analytic": bool(getattr(args, "postprocess_analytic", True)),
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
            pass
        if ckpt.is_dir() and (ckpt / "ar.ckpt").is_file():
            return ckpt

    if args.exp_dir:
        ckpt_dir = Path(args.exp_dir) / "checkpoints"
        if (ckpt_dir / "ar.ckpt").is_file():
            return ckpt_dir

    default = Path("/data/hdd/outputs/AutoBrep")
    if default.is_dir() and (default / "ar.ckpt").is_file():
        return default

    raise FileNotFoundError(
        "need --weight-folder with ar.ckpt + FSQ ckpts "
        "(condition Lightning ckpt is separate via --checkpoint)"
    )


def sample_batch(
    config: DotDict,
    model: AutoRegressiveSampler,
    *,
    infer_dir: Path,
    debug_dir: Path | None,
    point_cloud: Any = None,
    images: Any = None,
    dxf: Optional[dict[str, Any]] = None,
    condition_meta: Optional[dict[str, Any]] = None,
    fixed_stem: str = "",
    predictions_dir: Path | None = None,
) -> dict[str, Any]:
    stem = fixed_stem or generate_random_string(20)
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

    dxf = dxf or {}
    samples = model.sample_tokens(
        config=config,
        batch_size=config.hyper_parameters.batch_size,
        point_cloud=point_cloud,
        images=images,
        prim_types=dxf.get("prim_types"),
        prim_linetypes=dxf.get("prim_linetypes"),
        prim_geom=dxf.get("prim_geom"),
        prim_mask=dxf.get("prim_mask"),
    )
    batch_decoded = model.decode_tokens(samples.detach().cpu().numpy())
    batch_cad_data = model.convert_to_cad_data(batch_decoded) if batch_decoded else []

    def _cond_flags() -> dict[str, Any]:
        return {
            "pc_conditioned": bool(getattr(model, "pc_conditioned", False)),
            "view_conditioned": bool(getattr(model, "view_conditioned", False)),
        }

    with timer("Time to rebuild: %s seconds"):
        if not batch_cad_data:
            sample_stem = stem if fixed_stem else f"{stem}_000"
            sidecar = {
                "sample_id": sample_stem,
                "complexity": config.hyper_parameters.complexity,
                **_cond_flags(),
                "decode_ok": False,
                "error": "decode_failed",
                "files": [],
            }
            if condition_meta:
                sidecar["condition"] = condition_meta
            (infer_dir / f"{sample_stem}.json").write_text(
                json.dumps(sidecar, indent=2), encoding="utf-8"
            )
            log["invalid"] += 1
            log["errors"].append({"sample": sample_stem, "error": "decode_failed"})
            return log

        for sample_idx, cad_data in enumerate(batch_cad_data):
            if fixed_stem and sample_idx == 0:
                sample_stem = fixed_stem
            else:
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
                    sidecar = {
                        "sample_id": sample_stem,
                        "complexity": config.hyper_parameters.complexity,
                        **_cond_flags(),
                        "decode_ok": True,
                        "error": "rebuild_failed",
                        "num_faces": int(cad_data.face_pos_cad.shape[0]),
                        "files": [],
                    }
                    if condition_meta:
                        sidecar["condition"] = condition_meta
                    (infer_dir / f"{sample_stem}.json").write_text(
                        json.dumps(sidecar, indent=2), encoding="utf-8"
                    )
                    log["errors"].append(
                        {"sample": sample_stem, "error": "rebuild_failed"}
                    )
                    continue

                if bool(getattr(config.hyper_parameters, "postprocess_analytic", True)):
                    try:
                        from autobrep.inference.step_postprocess import postprocess_shape

                        result, _ = postprocess_shape(
                            result,
                            analytic=True,
                            sew_tolerance=max(
                                float(config.hyper_parameters.sewing_tolerance), 0.005
                            ),
                        )
                    except Exception as exc:  # noqa: BLE001
                        log.setdefault("postprocess_errors", []).append(
                            {"sample": sample_stem, "error": str(exc)}
                        )

                step_path = infer_dir / f"{sample_stem}.step"
                save_step_func([result], step_path)
                if predictions_dir is not None and fixed_stem and sample_idx == 0:
                    predictions_dir.mkdir(parents=True, exist_ok=True)
                    save_step_func([result], predictions_dir / f"{fixed_stem}.step")
                sidecar = {
                    "sample_id": sample_stem,
                    "complexity": config.hyper_parameters.complexity,
                    "temperature": config.hyper_parameters.temperature,
                    "top_p": config.hyper_parameters.sample_method.top_p_threshold,
                    **_cond_flags(),
                    "decode_ok": True,
                    "num_faces": int(cad_data.face_pos_cad.shape[0]),
                    "files": [f"infer/{sample_stem}.step"],
                }
                if condition_meta:
                    sidecar["condition"] = condition_meta
                (infer_dir / f"{sample_stem}.json").write_text(
                    json.dumps(sidecar, indent=2), encoding="utf-8"
                )
                log["valid"] += 1
                log["files"].append(f"infer/{sample_stem}.step")
            except Exception as exc:  # noqa: BLE001
                log["invalid"] += 1
                log["errors"].append({"sample": sample_stem, "error": str(exc)})

    return log


def write_manifest(
    output_dir: Path,
    *,
    dataset: str,
    task: str,
    config: DotDict,
    batch_logs: list[dict[str, Any]],
    pc_conditioned: bool,
    view_conditioned: bool = False,
    pc_ckpt: Optional[str] = None,
    view_ckpt: Optional[str] = None,
    condition_meta: Optional[dict[str, Any]] = None,
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
    if view_conditioned:
        mode = "view-cond"
    elif pc_conditioned:
        mode = "pc-cond"
    else:
        mode = "uncond"
    payload = {
        "dataset": dataset,
        "task": task,
        "repo": "AutoBrep",
        "mode": mode,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": {
            "weight_folder": config.weight_folder,
            "pc_ckpt": pc_ckpt,
            "view_ckpt": view_ckpt,
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
            "pc_conditioned": pc_conditioned,
            "view_conditioned": view_conditioned,
            "condition": condition_meta,
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
    p.add_argument(
        "--checkpoint",
        default="",
        help="Condition Lightning ckpt or train run dir (PC / ECCV view)",
    )
    p.add_argument("--weight-folder", default="", help="Pretrained ar.ckpt + FSQ folder")
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
    p.add_argument(
        "--postprocess-analytic",
        type=int,
        default=1,
        help="Replace near-analytic BSpline faces + ShapeFix before STEP write (1=on)",
    )
    p.add_argument("--seed", type=int, default=689447)
    p.add_argument("--use-seed", type=_parse_bool, default=False)
    p.add_argument("--debug", type=_parse_bool, default=True)
    p.add_argument("--format", default="step")
    p.add_argument("--index", default="")
    p.add_argument("--sample-id", default="")
    p.add_argument("--resume-from", default="")
    p.add_argument("--pc-conditioned", type=_parse_bool, default=False)
    p.add_argument("--view-conditioned", type=_parse_bool, default=False)
    p.add_argument("--point-cloud", default="", help=".npy (N,3) point cloud")
    p.add_argument("--abc-stem", default="", help="Sample PC from ABC-1M stem")
    p.add_argument("--abc-split", default="train", choices=["train", "val", "test"])
    p.add_argument("--pc-num-points", type=int, default=2048)
    p.add_argument(
        "--infer-split-name",
        default="val",
        choices=["train", "val", "test", "public_test"],
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.sample_id or args.view_conditioned:
        args.view_conditioned = True
    if args.point_cloud or args.abc_stem or args.pc_conditioned:
        args.pc_conditioned = True
    if args.view_conditioned and args.pc_conditioned:
        raise ValueError("Choose either --view-conditioned or --pc-conditioned, not both")

    weight_folder = resolve_weight_folder(args)
    args.weight_folder = str(weight_folder)

    output_dir = Path(args.output_dir)
    infer_dir = output_dir / "infer"
    infer_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = infer_dir / "debug" if args.debug else None
    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir = output_dir / "predictions" if args.view_conditioned else None

    if args.gpu != "":
        import os

        os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))

    config = build_sample_config(args)
    checkpoints = ARGenCheckpointPaths.from_folder(folder=str(weight_folder))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("AutoBrep infer requires CUDA")

    pc_ckpt: Optional[str] = None
    view_ckpt: Optional[str] = None
    if args.view_conditioned:
        view_path = resolve_view_checkpoint(
            checkpoint=args.checkpoint,
            exp_dir=args.exp_dir,
            weight_folder="",
        )
        view_ckpt = str(view_path)
        print(f"[AutoBrep infer] view_ckpt={view_ckpt}", flush=True)
    elif args.pc_conditioned:
        pc_path = resolve_pc_checkpoint(
            checkpoint=args.checkpoint,
            exp_dir=args.exp_dir,
            weight_folder="",
        )
        pc_ckpt = str(pc_path)
        print(f"[AutoBrep infer] pc_ckpt={pc_ckpt}", flush=True)

    model = AutoRegressiveSampler(
        checkpoint_paths=checkpoints,
        device=device,
        pc_conditioned=bool(args.pc_conditioned),
        pc_ckpt=pc_ckpt,
        view_conditioned=bool(args.view_conditioned),
        view_ckpt=view_ckpt,
    )

    point_cloud = None
    images = None
    dxf: dict[str, Any] = {}
    condition_meta: dict[str, Any] = {}
    fixed_stem = ""

    if args.view_conditioned:
        data_root = args.data_dir or "/data/hdd/datasets/eccv2026ws-cad-data"
        sample_id = args.sample_id or args.index
        if not sample_id:
            import json as _json

            split_path = Path(data_root) / "processed" / "datasplit.json"
            if not split_path.is_file():
                split_path = Path(data_root) / "processed" / "autobrep" / "datasplit.json"
            split_data = _json.loads(split_path.read_text(encoding="utf-8"))
            ids = split_data["splits"].get(args.infer_split_name) or []
            if not ids:
                raise ValueError(
                    f"No sample ids in split={args.infer_split_name}; "
                    "pass --sample-id explicitly"
                )
            sample_id = ids[0]
            print(
                f"[AutoBrep infer] no --sample-id; using first of "
                f"{args.infer_split_name}: {sample_id}",
                flush=True,
            )
        fixed_stem = str(sample_id)
        images, dxf, condition_meta = load_condition_for_sample(
            data_root,
            sample_id,
            batch_size=args.batch_size,
        )
        condition_meta["view_ckpt"] = view_ckpt
        condition_meta["infer_split_name"] = args.infer_split_name
        print(
            f"[AutoBrep infer] condition sample_id={sample_id} "
            f"images={tuple(images.shape)} n_prims={condition_meta.get('n_prims')}",
            flush=True,
        )

    elif args.pc_conditioned:
        rng_seed = int(args.seed) if args.use_seed else 0
        if args.point_cloud:
            point_cloud, condition_meta = load_point_cloud_npy(
                args.point_cloud,
                batch_size=args.batch_size,
                num_points=args.pc_num_points,
                rng=__import__("numpy").random.default_rng(rng_seed),
            )
        elif args.abc_stem:
            data_root = args.data_dir or "/data/hdd/datasets/ABC-1M"
            point_cloud, condition_meta = load_point_cloud_from_abc(
                data_root,
                args.abc_stem,
                batch_size=args.batch_size,
                num_points=args.pc_num_points,
                split=args.abc_split,
                seed=rng_seed,
            )
        else:
            raise ValueError(
                "--pc-conditioned requires --point-cloud /path.npy "
                "or --abc-stem <stem> (with --data-dir ABC-1M)"
            )
        pc_npy = infer_dir / "condition_point_cloud.npy"
        pc_png = infer_dir / "condition_point_cloud.png"
        pts_np = point_cloud[0].detach().cpu().numpy()
        save_point_cloud_npy(pc_npy, pts_np)
        title = condition_meta.get("stem") or condition_meta.get("path") or "point cloud"
        save_point_cloud_preview(pc_png, pts_np, title=str(title))
        condition_meta["saved_npy"] = f"infer/{pc_npy.name}"
        condition_meta["saved_preview"] = f"infer/{pc_png.name}"
        condition_meta["pc_ckpt"] = pc_ckpt
        condition_meta["pc_num_points"] = int(point_cloud.shape[1])
        print(
            f"[AutoBrep infer] condition PC N={point_cloud.shape[1]} "
            f"preview={pc_png}",
            flush=True,
        )

    if config.seed.use_seed:
        seed_everything(config.seed.seed_value, workers=True)

    print(
        f"[AutoBrep infer] weight_folder={weight_folder} "
        f"pc={args.pc_conditioned} view={args.view_conditioned} "
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
                images=images,
                dxf=dxf,
                condition_meta=condition_meta or None,
                fixed_stem=fixed_stem if args.view_conditioned else "",
                predictions_dir=predictions_dir,
            )
        )
    write_manifest(
        output_dir,
        dataset=args.dataset,
        task=args.task,
        config=config,
        batch_logs=batch_logs,
        pc_conditioned=bool(args.pc_conditioned),
        view_conditioned=bool(args.view_conditioned),
        pc_ckpt=pc_ckpt,
        view_ckpt=view_ckpt,
        condition_meta=condition_meta or None,
    )
    print(f"[AutoBrep infer] done in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
