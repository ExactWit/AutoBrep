#!/usr/bin/env python3
"""Official ECCV test for classical Wesley–Markowsky orthographic baseline (no DL)."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import shutil
import time
from pathlib import Path
from typing import Any

import pyarrow.dataset as ds

from autobrep.classical.orthographic_wm import reconstruct_from_techdraw_paths
from autobrep.inference.eccv_val_eval import (
    DEFAULT_EVAL_PY,
    load_official_val_ids,
    prepare_gt_pred_dirs,
    run_official_eval,
)


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _load_path_map(parquet_root: Path, split: str) -> dict[str, dict[str, str]]:
    pq = parquet_root / split
    table = ds.dataset(str(pq), format="parquet").to_table(
        columns=["sample_id", "techdraw_dxf_path", "techdraw_svg_path"]
    )
    out: dict[str, dict[str, str]] = {}
    ids = table.column("sample_id").to_pylist()
    dxfs = table.column("techdraw_dxf_path").to_pylist()
    svgs = table.column("techdraw_svg_path").to_pylist()
    for sid, dxf, svg in zip(ids, dxfs, svgs):
        out[str(sid)] = {
            "techdraw_dxf_path": str(dxf or ""),
            "techdraw_svg_path": str(svg or ""),
        }
    return out


def _worker_reconstruct(payload: dict[str, Any], conn) -> None:
    """Child process target — isolate OCC / combinatorial hangs."""
    try:
        res = reconstruct_from_techdraw_paths(
            payload["data_dir"],
            dxf_rel=payload["dxf_rel"],
            svg_rel=payload["svg_rel"],
            out_step=payload["out_step"],
            tol=payload["tol"],
            min_score=float(payload["min_score"]),
        )
        conn.send(
            {
                "ok": bool(res.ok),
                "error": res.error,
                "n_verts": res.n_verts,
                "n_edges": res.n_edges,
                "n_faces": res.n_faces,
                "score": res.score,
                "method": res.method,
            }
        )
    except Exception as exc:  # noqa: BLE001
        conn.send({"ok": False, "error": str(exc)})
    finally:
        conn.close()


def _reconstruct_with_timeout(
    *,
    data_dir: Path,
    dxf_rel: str,
    svg_rel: str,
    out_step: Path,
    tol: float | None,
    min_score: float,
    timeout_sec: float,
) -> dict[str, Any]:
    if timeout_sec <= 0:
        res = reconstruct_from_techdraw_paths(
            data_dir,
            dxf_rel=dxf_rel,
            svg_rel=svg_rel,
            out_step=out_step,
            tol=tol,
            min_score=min_score,
        )
        return {
            "ok": bool(res.ok and out_step.is_file()),
            "error": res.error,
            "n_verts": res.n_verts,
            "n_edges": res.n_edges,
            "n_faces": res.n_faces,
            "score": res.score,
            "method": res.method,
        }

    ctx = mp.get_context("spawn")
    parent, child = ctx.Pipe(duplex=False)
    proc = ctx.Process(
        target=_worker_reconstruct,
        args=(
            {
                "data_dir": str(data_dir),
                "dxf_rel": dxf_rel,
                "svg_rel": svg_rel,
                "out_step": str(out_step),
                "tol": tol,
                "min_score": min_score,
            },
            child,
        ),
    )
    proc.start()
    child.close()
    proc.join(timeout_sec)
    if proc.is_alive():
        proc.terminate()
        proc.join(5)
        if proc.is_alive():
            proc.kill()
            proc.join(5)
        out_step.unlink(missing_ok=True)
        return {
            "ok": False,
            "error": f"timeout>{timeout_sec:.0f}s",
            "n_verts": 0,
            "n_edges": 0,
            "n_faces": 0,
            "score": 0.0,
            "method": "",
        }
    if parent.poll():
        status = parent.recv()
    else:
        status = {"ok": False, "error": f"worker_exit={proc.exitcode}"}
    parent.close()
    status["ok"] = bool(status.get("ok") and out_step.is_file())
    return status


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Classical orthographic WM official test")
    p.add_argument("--exp-dir", required=True)
    p.add_argument("--output-dir", default="")
    p.add_argument("--data-dir", default="/data/hdd/datasets/eccv2026ws-cad-data")
    p.add_argument("--datasplit", default="")
    p.add_argument("--split", default="test")
    p.add_argument("--eval-py", default=str(DEFAULT_EVAL_PY))
    p.add_argument("--limit-samples", type=int, default=-1)
    p.add_argument("--tol", type=float, default=-1.0)
    p.add_argument("--min-score", type=float, default=0.05)
    p.add_argument(
        "--sample-timeout",
        type=float,
        default=120.0,
        help="Per-sample wall timeout seconds (0=disable). Default 120.",
    )
    p.add_argument(
        "--resume-pred",
        default="",
        help="Copy existing STEP preds from this dir then skip those sample ids",
    )
    p.add_argument("--make-submission-zip", type=_parse_bool, default=False)
    return p.parse_args()

def main() -> None:
    args = parse_args()
    exp_dir = Path(args.exp_dir)
    data_dir = Path(args.data_dir)
    # Launcher may pass .../processed; TechDraw + GT live under dataset root.
    if data_dir.name == "processed" and (data_dir.parent / "train").is_dir():
        data_dir = data_dir.parent
    out_dir = Path(args.output_dir) if args.output_dir else exp_dir
    metrics_dir = exp_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    datasplit = Path(args.datasplit) if args.datasplit else (
        data_dir / "processed" / "datasplit.json"
    )
    parquet_root = data_dir / "processed" / "autobrep"
    ids = load_official_val_ids(
        data_dir,
        max_samples=int(args.limit_samples),
        split=str(args.split),
        datasplit=datasplit,
        parquet_root=parquet_root,
        require_gt=str(args.split) != "public_test",
    )
    path_map = _load_path_map(parquet_root, str(args.split))
    work = metrics_dir / f"classical_wm_{args.split}"
    gt_dir, pred_dir, ok_ids = prepare_gt_pred_dirs(data_dir, ids, work)

    resume_pred = Path(args.resume_pred) if str(args.resume_pred).strip() else None
    resumed = 0
    if resume_pred is not None and resume_pred.is_dir():
        for src in resume_pred.glob("*.step"):
            dst = pred_dir / src.name
            if not dst.is_file():
                shutil.copy2(src, dst)
                resumed += 1
        print(
            f"[classical_wm] resumed {resumed} STEP(s) from {resume_pred}",
            flush=True,
        )

    print(
        f"[classical_wm] split={args.split} n_ids={len(ids)} with_gt={len(ok_ids)} "
        f"timeout={float(args.sample_timeout):.0f}s → {work}",
        flush=True,
    )

    gen_log: list[dict[str, Any]] = []
    t0 = time.time()
    tol = None if float(args.tol) < 0 else float(args.tol)
    timeout_sec = float(args.sample_timeout)
    for i, sid in enumerate(ok_ids):
        row = path_map.get(str(sid), {})
        out_step = pred_dir / f"{sid}.step"
        if out_step.is_file() and out_step.stat().st_size > 0:
            status = {
                "sample_id": sid,
                "ok": True,
                "error": "",
                "n_verts": 0,
                "n_edges": 0,
                "n_faces": 0,
                "score": 0.0,
                "method": "resumed",
            }
        else:
            try:
                status = _reconstruct_with_timeout(
                    data_dir=data_dir,
                    dxf_rel=row.get("techdraw_dxf_path", ""),
                    svg_rel=row.get("techdraw_svg_path", ""),
                    out_step=out_step,
                    tol=tol,
                    min_score=float(args.min_score),
                    timeout_sec=timeout_sec,
                )
                status = {
                    "sample_id": sid,
                    "ok": bool(status.get("ok")),
                    "error": status.get("error", ""),
                    "n_verts": status.get("n_verts", 0),
                    "n_edges": status.get("n_edges", 0),
                    "n_faces": status.get("n_faces", 0),
                    "score": status.get("score", 0.0),
                    "method": status.get("method", ""),
                }
            except Exception as exc:  # noqa: BLE001
                status = {"sample_id": sid, "ok": False, "error": str(exc)}
                out_step.unlink(missing_ok=True)
        gen_log.append(status)
        # Log every sample so hangs are visible immediately.
        print(
            f"[classical_wm] [{i+1}/{len(ok_ids)}] {sid}: ok={status.get('ok')} "
            f"method={status.get('method','')} score={float(status.get('score') or 0):.3f} "
            f"err={status.get('error','')}",
            flush=True,
        )
    n_ok = sum(1 for g in gen_log if g.get("ok"))
    for sid in list(ok_ids):
        if not (pred_dir / f"{sid}.step").is_file():
            (gt_dir / f"{sid}.step").unlink(missing_ok=True)

    metrics: dict[str, Any] = {
        "n_requested": len(ok_ids),
        "n_generated": n_ok,
        "gen_success_rate": n_ok / max(len(ok_ids), 1),
        "elapsed_sec": time.time() - t0,
        "method": "wesley_markowsky_classical",
        "gen_log": gen_log,
    }
    pred_steps = list(pred_dir.glob("*.step"))
    if pred_steps and str(args.split) != "public_test":
        try:
            official = run_official_eval(work, eval_py=Path(args.eval_py))
            metrics["official"] = {
                k: v for k, v in official.items() if not k.startswith("_")
            }
            metrics["official_log_tail"] = official.get("_raw_log_tail", "")[-2000:]
        except Exception as exc:  # noqa: BLE001
            print(f"[classical_wm] official eval failed: {exc}", flush=True)
            metrics["official_error"] = str(exc)
            metrics["official"] = {
                "summary": 0.0,
                "valid_ratio": 0.0,
                "surface_f1": 0.0,
                "edge_f1": 0.0,
                "vertex_f1": 0.0,
                "topo_f1": 0.0,
            }
    else:
        metrics["official"] = {
            "summary": 0.0,
            "valid_ratio": 0.0,
            "surface_f1": 0.0,
            "edge_f1": 0.0,
            "vertex_f1": 0.0,
            "topo_f1": 0.0,
        }

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
        "elapsed_sec": metrics.get("elapsed_sec"),
        "checkpoint": "classical_wm",
        "work_dir": str(work),
        "official": off,
    }
    (metrics_dir / "test.json").write_text(
        json.dumps(test_json, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (metrics_dir / "classical_wm_gen_log.json").write_text(
        json.dumps(gen_log, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"[classical_wm] done gen={n_ok}/{len(ok_ids)} "
        f"summary={test_json['summary']:.4f} valid={test_json['valid_ratio']:.4f} "
        f"→ {metrics_dir / 'test.json'}",
        flush=True,
    )

    if args.make_submission_zip:
        import zipfile

        zip_path = out_dir / "submission.zip"
        steps = sorted(pred_dir.glob("*.step"))
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for step in steps:
                zf.write(step, arcname=f"predictions/{step.name}")
        print(f"[classical_wm] submission.zip n={len(steps)} → {zip_path}", flush=True)


if __name__ == "__main__":
    main()
