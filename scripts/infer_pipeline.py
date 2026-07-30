#!/usr/bin/env python3
"""exp_launcher infer entry: unconditional + BOGEOM autocomplete → STEP."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Optional

import torch
from pytorch_lightning import seed_everything

from autobrep.inference.autocomplete_prompt import (
    AutocompleteVocab,
    build_geom_tokens_from_arrays,
    find_abc_row_by_stem,
    load_condition_from_abc_row,
    load_condition_from_json,
)
from autobrep.inference.brepgen_brep_builder import AutoBrepBuilder
from autobrep.inference.inference_common import (
    reconstruct_compound,
    save_debug_images,
    save_point_grid,
)
from autobrep.models.autoregressive import ARGenCheckpointPaths, AutoRegressiveSampler
from autobrep.models.vaes import EdgeFSQVAE, SurfaceFSQVAE
from autobrep.utils import DotDict, generate_random_string, timer
from occwl.io import save_step as save_step_func


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on", "autocomplete"}


def _parse_face_ids(raw: str) -> Optional[list[int]]:
    raw = (raw or "").strip()
    if not raw:
        return None
    return [int(x.strip()) for x in raw.split(",") if x.strip() != ""]


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


def load_encode_fsq(weight_folder: Path, device: torch.device):
    """FSQ with encoders kept (sampler drops encoder for decode-only)."""
    surf = SurfaceFSQVAE.load_from_checkpoint(
        str(weight_folder / "surf-fsq.ckpt"),
        map_location="cpu",
        use_dcae=True,
    ).to(device).eval()
    edge = EdgeFSQVAE.load_from_checkpoint(
        str(weight_folder / "edge-fsq.ckpt"),
        map_location="cpu",
        use_dcae=True,
    ).to(device).eval()
    return surf, edge


def vocab_from_sampler(model: AutoRegressiveSampler) -> AutocompleteVocab:
    hp = model.transformer.hparams
    return AutocompleteVocab(
        bit=int(hp.bit),
        max_face=int(hp.max_face),
        surf_codebook_size=int(hp.surf_codebook_size),
        edge_codebook_size=int(hp.edge_codebook_size),
    )


def resolve_condition(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return (condition arrays dict, sidecar meta)."""
    if args.condition_json:
        cond = load_condition_from_json(args.condition_json)
        return cond, {"source": "json", **cond.get("meta", {})}

    if not args.abc_stem and not args.autocomplete:
        return {}, {}

    data_root = args.data_dir or "/data/hdd/datasets/ABC-1M"
    if not args.abc_stem:
        raise ValueError(
            "autocomplete without --condition-json requires --abc-stem "
            "(or use scripts/autocomplete_smoke.py)"
        )
    row = find_abc_row_by_stem(
        data_root, args.abc_stem, split=args.abc_split
    )
    face_ids = _parse_face_ids(args.face_ids)
    num_faces = int(args.num_condition_faces) if args.num_condition_faces else None
    cond = load_condition_from_abc_row(
        row,
        face_ids=face_ids,
        num_faces=num_faces,
        mode=args.condition_mode,
    )
    return cond, {"source": "abc", **cond.get("meta", {})}


def build_geom_prompt(
    cond: dict[str, Any],
    *,
    surface_fsq,
    edge_fsq,
    device: torch.device,
    vocab: AutocompleteVocab,
) -> tuple[list[int], list[int]]:
    return build_geom_tokens_from_arrays(
        cond["face_pos"],
        cond["face_ncs"],
        cond["edge_pos"],
        cond["edge_ncs"],
        cond["face_edge_adj"],
        cond["user_face_indices"],
        surface_fsq=surface_fsq,
        edge_fsq=edge_fsq,
        device=device,
        vocab=vocab,
    )


def sample_batch(
    config: DotDict,
    model: AutoRegressiveSampler,
    *,
    infer_dir: Path,
    debug_dir: Path | None,
    geom_prompt_tokens: Optional[list[int]] = None,
    condition_meta: Optional[dict[str, Any]] = None,
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
        geom_prompt_tokens=geom_prompt_tokens,
    )
    if geom_prompt_tokens is not None:
        prompt_len = 4 + len(geom_prompt_tokens) + 1
    else:
        prompt_len = 5

    batch_np = samples.detach().cpu().numpy()
    batch_decoded = model.decode_tokens(batch_np)
    decode_fallback = False
    # Zero-shot BOGEOM often yields malformed CAD tokens; fall back to condition geom only.
    if not batch_decoded and geom_prompt_tokens is not None:
        from autobrep.data.token_mapping import MMTokenIndex
        import numpy as np

        geom = list(geom_prompt_tokens)
        # BOC..EOC empty → decode_tokens merges geom_tokens as the CAD body
        synth = np.array(
            [
                MMTokenIndex.BOS.value,
                *geom,
                MMTokenIndex.BOC.value,
                MMTokenIndex.EOC.value,
                MMTokenIndex.EOS.value,
            ],
            dtype=np.int64,
        )
        batch_decoded = model.decode_tokens(synth[None, :])
        decode_fallback = bool(batch_decoded)
        if decode_fallback:
            print(
                "[AutoBrep infer] full-sample decode failed; "
                "rebuilt from condition BOGEOM only",
                flush=True,
            )

    batch_cad_data = model.convert_to_cad_data(batch_decoded) if batch_decoded else []

    with timer("Time to rebuild: %s seconds"):
        if not batch_cad_data:
            sample_stem = f"{stem}_000"
            sidecar = {
                "sample_id": sample_stem,
                "complexity": config.hyper_parameters.complexity,
                "temperature": config.hyper_parameters.temperature,
                "top_p": config.hyper_parameters.sample_method.top_p_threshold,
                "autocomplete": geom_prompt_tokens is not None,
                "prompt_len": prompt_len,
                "decode_ok": False,
                "files": [],
                "error": "decode_failed",
            }
            if condition_meta:
                sidecar.update(condition_meta)
            (infer_dir / f"{sample_stem}.json").write_text(
                json.dumps(sidecar, indent=2), encoding="utf-8"
            )
            log["invalid"] += 1
            log["errors"].append({"sample": sample_stem, "error": "decode_failed"})
            return log

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
                    sidecar = {
                        "sample_id": sample_stem,
                        "autocomplete": geom_prompt_tokens is not None,
                        "prompt_len": prompt_len,
                        "decode_ok": True,
                        "decode_fallback_geom_only": decode_fallback,
                        "error": "rebuild_failed",
                        "files": [],
                    }
                    if condition_meta:
                        sidecar.update(condition_meta)
                    (infer_dir / f"{sample_stem}.json").write_text(
                        json.dumps(sidecar, indent=2), encoding="utf-8"
                    )
                    log["errors"].append(
                        {"sample": sample_stem, "error": "rebuild_failed"}
                    )
                    continue

                step_path = infer_dir / f"{sample_stem}.step"
                save_step_func([result], step_path)
                num_faces = int(cad_data.face_pos_cad.shape[0])
                sidecar = {
                    "sample_id": sample_stem,
                    "complexity": config.hyper_parameters.complexity,
                    "temperature": config.hyper_parameters.temperature,
                    "top_p": config.hyper_parameters.sample_method.top_p_threshold,
                    "autocomplete": geom_prompt_tokens is not None,
                    "prompt_len": prompt_len,
                    "num_faces": num_faces,
                    "decode_ok": True,
                    "decode_fallback_geom_only": decode_fallback,
                    "files": [f"infer/{sample_stem}.step"],
                }
                if condition_meta:
                    sidecar.update(condition_meta)
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
    autocomplete: bool,
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
        "mode": "autocomplete" if autocomplete else "uncond",
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
            "autocomplete": autocomplete,
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
    # Autocomplete
    p.add_argument("--autocomplete", type=_parse_bool, default=False)
    p.add_argument("--condition-json", default="")
    p.add_argument("--abc-stem", default="")
    p.add_argument("--face-ids", default="")
    p.add_argument("--num-condition-faces", default="")
    p.add_argument(
        "--condition-mode",
        default="random",
        choices=["random", "constraint"],
    )
    p.add_argument("--abc-split", default="train", choices=["train", "val", "test"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    weight_folder = resolve_weight_folder(args)
    args.weight_folder = str(weight_folder)

    # Enable autocomplete if any condition source is given
    if args.condition_json or args.abc_stem:
        args.autocomplete = True

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

    model = AutoRegressiveSampler(
        checkpoint_paths=checkpoints,
        device=device,
    )

    geom_prompt_tokens: Optional[list[int]] = None
    condition_meta: dict[str, Any] = {}
    if args.autocomplete:
        cond, condition_meta = resolve_condition(args)
        if not cond:
            raise ValueError("autocomplete enabled but no condition resolved")
        surf_enc, edge_enc = load_encode_fsq(weight_folder, device)
        vocab = vocab_from_sampler(model)
        geom_prompt_tokens, remapped = build_geom_prompt(
            cond,
            surface_fsq=surf_enc,
            edge_fsq=edge_enc,
            device=device,
            vocab=vocab,
        )
        condition_meta = {
            **condition_meta,
            "face_ids": cond["user_face_indices"],
            "remapped_face_ids": remapped,
            "geom_token_len": len(geom_prompt_tokens),
            "prompt_len": 4 + len(geom_prompt_tokens) + 1,
        }
        print(
            f"[AutoBrep infer] autocomplete geom_tokens={len(geom_prompt_tokens)} "
            f"faces={cond['user_face_indices']}",
            flush=True,
        )
        del surf_enc, edge_enc
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if config.seed.use_seed:
        seed_everything(config.seed.seed_value, workers=True)

    print(
        f"[AutoBrep infer] weight_folder={weight_folder} "
        f"autocomplete={args.autocomplete} "
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
                geom_prompt_tokens=geom_prompt_tokens,
                condition_meta=condition_meta,
            )
        )
    write_manifest(
        output_dir,
        dataset=args.dataset,
        task=args.task,
        config=config,
        batch_logs=batch_logs,
        autocomplete=bool(args.autocomplete),
    )
    print(f"[AutoBrep infer] done in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
