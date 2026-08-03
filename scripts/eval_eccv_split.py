#!/usr/bin/env python3
"""Official ECCV split eval: test (min_eval metrics) / public_test (submission.zip)."""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from typing import Any

import torch

from autobrep.inference.eccv_val_eval import (
    DEFAULT_EVAL_PY,
    generate_pred_steps_batched,
    load_official_val_ids,
    prepare_gt_pred_dirs,
    run_official_eval,
)
from autobrep.inference.view_condition import resolve_view_checkpoint
from autobrep.models.autoregressive import ARGenCheckpointPaths, AutoBrepViewModel
from autobrep.models.vaes import EdgeFSQVAE, SurfaceFSQVAE


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _strip_thop(model: torch.nn.Module) -> None:
    for mod in model.modules():
        for name in ("total_ops", "total_params"):
            if name in getattr(mod, "_buffers", {}):
                del mod._buffers[name]


def _write_test_metrics(
    *,
    exp_dir: Path,
    work: Path,
    metrics: dict[str, Any],
    sample_ids: list[str],
    gen_log: list[dict[str, Any]],
    checkpoint: str,
) -> None:
    metrics_dir = exp_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    off = metrics.get("official") or {}
    test_json = {
        "summary": float(off.get("summary", 0.0)),
        "valid_ratio": float(off.get("valid_ratio", 0.0)),
        "surface_f1": float(off.get("surface_f1", 0.0)),
        "edge_f1": float(off.get("edge_f1", 0.0)),
        "vertex_f1": float(off.get("vertex_f1", 0.0)),
        "topo_f1": float(off.get("topo_f1", 0.0)),
        "n_requested": metrics.get("n_requested"),
        "n_generated": metrics.get("n_generated"),
        "gen_success_rate": metrics.get("gen_success_rate"),
        "checkpoint": checkpoint,
        "work_dir": str(work),
        "official": off,
    }
    (metrics_dir / "test.json").write_text(
        json.dumps(test_json, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    per_sample = []
    by_id = {str(g.get("sample_id")): g for g in gen_log}
    for i, sid in enumerate(sample_ids):
        g = by_id.get(str(sid), {})
        per_sample.append(
            {
                "index": i,
                "sample_id": sid,
                "stem": sid,
                "ok": bool(g.get("ok")),
                "error": g.get("error", ""),
                "summary": float(off.get("summary", 0.0)) if g.get("ok") else None,
                "has_metrics": bool(g.get("ok")),
            }
        )
    (metrics_dir / "test_per_sample.json").write_text(
        json.dumps(
            {
                "metric": "summary",
                "sort": "desc",
                "task": "gen",
                "checkpoint": checkpoint,
                "test": per_sample,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[eval_eccv] wrote {metrics_dir / 'test.json'}", flush=True)


def _make_submission_zip(pred_dir: Path, zip_path: Path) -> int:
    steps = sorted(pred_dir.glob("*.step"))
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for step in steps:
            zf.write(step, arcname=f"predictions/{step.name}")
    print(f"[eval_eccv] submission.zip n={len(steps)} → {zip_path}", flush=True)
    return len(steps)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ECCV official test / public_test STEP eval")
    p.add_argument("--exp-dir", required=True)
    p.add_argument("--output-dir", default="")
    p.add_argument("--data-dir", default="/data/hdd/datasets/eccv2026ws-cad-data")
    p.add_argument("--datasplit", default="")
    p.add_argument("--dataset", default="eccv2026ws-cad-data")
    p.add_argument("--task", default="gen")
    p.add_argument("--checkpoint", default="")
    p.add_argument("--weight-folder", default="/data/hdd/outputs/AutoBrep")
    p.add_argument("--gpu", default="0")
    p.add_argument(
        "--split",
        default="test",
        choices=["val", "test", "public_test"],
    )
    p.add_argument("--max-samples", type=int, default=-1)
    p.add_argument("--gen-batch", type=int, default=1)
    p.add_argument("--gen-retries", type=int, default=1)
    p.add_argument(
        "--gen-rerank",
        type=_parse_bool,
        default=False,
        help="With gen-retries>1: sample all K and keep best success",
    )
    p.add_argument("--complexity", default="from_condition")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--eval-py", default=str(DEFAULT_EVAL_PY))
    p.add_argument("--make-submission-zip", type=_parse_bool, default=False)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.gpu != "":
        import os

        os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))

    exp_dir = Path(args.exp_dir)
    output_dir = Path(args.output_dir) if args.output_dir else exp_dir
    data_root = Path(args.data_dir)
    datasplit = (
        Path(args.datasplit)
        if args.datasplit
        else data_root / "processed" / "datasplit.json"
    )
    split = str(args.split)
    is_public = split == "public_test"

    ckpt = resolve_view_checkpoint(
        checkpoint=args.checkpoint,
        exp_dir=str(exp_dir),
        weight_folder="",
    )
    print(f"[eval_eccv] split={split} ckpt={ckpt}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("eval_eccv_split requires CUDA")

    model = AutoBrepViewModel.load_from_checkpoint(
        str(ckpt), map_location="cpu", strict=False
    )
    _strip_thop(model)
    model = model.to(device).eval()

    paths = ARGenCheckpointPaths.from_folder(str(args.weight_folder))
    surf = (
        SurfaceFSQVAE.load_from_checkpoint(paths.surface_fsq)
        .drop_encoder()
        .to(device)
        .eval()
    )
    edge = (
        EdgeFSQVAE.load_from_checkpoint(paths.edge_fsq).drop_encoder().to(device).eval()
    )

    ids = load_official_val_ids(
        data_root,
        max_samples=int(args.max_samples),
        split=split,
        datasplit=datasplit if datasplit.is_file() else None,
        parquet_root=None if is_public else (data_root / "processed" / "autobrep"),
        require_gt=not is_public,
    )
    if not ids:
        raise RuntimeError(f"no ids for split={split}")
    print(f"[eval_eccv] n_ids={len(ids)} head={ids[:6]}", flush=True)

    work = exp_dir / "metrics" / f"official_{split}"
    work.mkdir(parents=True, exist_ok=True)
    if is_public:
        pred_dir = output_dir / "predictions"
        if pred_dir.exists():
            import shutil

            shutil.rmtree(pred_dir)
        pred_dir.mkdir(parents=True, exist_ok=True)
        ok_ids = list(ids)
        gt_dir = work / "gt"
        gt_dir.mkdir(parents=True, exist_ok=True)
    else:
        gt_dir, pred_dir, ok_ids = prepare_gt_pred_dirs(data_root, ids, work)
        if not ok_ids:
            raise RuntimeError("no GT STEPs found for test/val")

    gen_log = generate_pred_steps_batched(
        transformer=model,
        surface_fsq=surf,
        edge_fsq=edge,
        device=device,
        dataset_root=data_root,
        sample_ids=ok_ids,
        pred_dir=pred_dir,
        complexity=str(args.complexity),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        gen_batch_size=max(1, int(args.gen_batch)),
        split=split,
        require_gt=not is_public,
        gen_retries=max(1, int(args.gen_retries)),
        gen_rerank=bool(args.gen_rerank),
    )
    for i, status in enumerate(gen_log):
        if (i + 1) % 10 == 0 or not status.get("ok") or i == 0 or i + 1 == len(gen_log):
            print(
                f"[eval_eccv] [{i+1}/{len(ok_ids)}] {status.get('sample_id')}: "
                f"ok={status.get('ok')} err={status.get('error', '')}",
                flush=True,
            )

    n_ok = sum(1 for g in gen_log if g.get("ok"))
    metrics: dict[str, Any] = {
        "split": split,
        "n_requested": len(ok_ids),
        "n_generated": n_ok,
        "gen_success_rate": n_ok / max(len(ok_ids), 1),
        "sample_ids": ok_ids,
        "gen_log": gen_log,
        "checkpoint": str(ckpt),
        "gen_batch": int(args.gen_batch),
        "gen_retries": int(args.gen_retries),
        "gen_rerank": bool(args.gen_rerank),
    }

    if not is_public:
        for sid in list(ok_ids):
            if not (pred_dir / f"{sid}.step").is_file():
                (gt_dir / f"{sid}.step").unlink(missing_ok=True)
        pred_steps = list(pred_dir.glob("*.step"))
        if pred_steps:
            try:
                official = run_official_eval(work, eval_py=Path(args.eval_py))
                metrics["official"] = {
                    k: v for k, v in official.items() if not k.startswith("_")
                }
            except Exception as exc:  # noqa: BLE001
                metrics["official_error"] = f"{type(exc).__name__}:{exc}"
                print(f"[eval_eccv] official eval failed: {exc}", flush=True)
        else:
            metrics["official"] = {
                "summary": 0.0,
                "valid_ratio": 0.0,
                "surface_f1": 0.0,
                "edge_f1": 0.0,
                "vertex_f1": 0.0,
                "topo_f1": 0.0,
            }
        (work / "metrics.json").write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        _write_test_metrics(
            exp_dir=exp_dir,
            work=work,
            metrics=metrics,
            sample_ids=ok_ids,
            gen_log=gen_log,
            checkpoint=str(ckpt),
        )
        off = metrics.get("official") or {}
        print(
            f"[eval_eccv] summary={float(off.get('summary', 0.0)):.4f} "
            f"surf={float(off.get('surface_f1', 0.0)):.4f} "
            f"gen_ok={n_ok}/{len(ok_ids)}",
            flush=True,
        )
    else:
        (work / "metrics.json").write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        (exp_dir / "metrics").mkdir(parents=True, exist_ok=True)
        (exp_dir / "metrics" / "public_infer.json").write_text(
            json.dumps(
                {
                    "n_requested": len(ok_ids),
                    "n_generated": n_ok,
                    "gen_success_rate": n_ok / max(len(ok_ids), 1),
                    "checkpoint": str(ckpt),
                    "predictions_dir": str(pred_dir),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        if args.make_submission_zip or True:
            _make_submission_zip(pred_dir, output_dir / "submission.zip")

    return 0 if n_ok > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
