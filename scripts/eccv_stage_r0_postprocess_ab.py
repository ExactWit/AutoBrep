#!/usr/bin/env python3
"""R0 gate: offline analytic STEP post-process A/B vs baseline GT test preds.

Zero training. Reuses existing official_test gt/ + pred/, writes analytic preds,
re-runs challenge min_eval, and emits a stage report markdown.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "core" / "src"))

from autobrep.inference.eccv_val_eval import DEFAULT_EVAL_PY, run_official_eval  # noqa: E402
from autobrep.inference.step_postprocess import postprocess_step_file  # noqa: E402

BASELINE_OFFICIAL = Path(
    "/data/hdd/exps/runs/eccv2026ws-cad-data/gen/AutoBrep/"
    "260725-002218/eccv-3view-geom-resume__test/metrics/official_test"
)
BASELINE_TEST_JSON = Path(
    "/data/hdd/exps/runs/eccv2026ws-cad-data/gen/AutoBrep/"
    "260725-002218/eccv-3view-geom-resume__test/metrics/test.json"
)
DEFAULT_OUT = Path(
    "/data/hdd/exps/runs/eccv2026ws-cad-data/gen/AutoBrep/"
    "stage_gates/R0_postprocess_ab"
)
DEFAULT_REPORT_DIR = REPO / "docs" / "eccv_stage_reports"


def _row_from_official(off: dict[str, Any], *, gen_n: int | None = None, n_req: int | None = None) -> dict[str, Any]:
    return {
        "summary": float(off.get("summary") or 0.0),
        "valid_ratio": float(off.get("valid_ratio") or 0.0),
        "invalid_ratio": float(
            off.get("invalid_ratio")
            if off.get("invalid_ratio") is not None
            else (1.0 - float(off.get("valid_ratio") or 0.0))
        ),
        "surface_f1": float(off.get("surface_f1") or 0.0),
        "edge_f1": float(off.get("edge_f1") or 0.0),
        "vertex_f1": float(off.get("vertex_f1") or 0.0),
        "topo_f1": float(off.get("topo_f1") or 0.0),
        "cd_surface": off.get("cd_surface"),
        "cd_edge": off.get("cd_edge"),
        "cd_vertex": off.get("cd_vertex"),
        "n_generated": gen_n,
        "n_requested": n_req,
        "gen_success_rate": (float(gen_n) / float(n_req)) if gen_n is not None and n_req else None,
    }


def _fmt(v: Any, digits: int = 6) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def _write_report(
    path: Path,
    *,
    raw: dict[str, Any],
    analytic: dict[str, Any],
    meta: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dsum = float(analytic["summary"]) - float(raw["summary"])
    lines = [
        "# R0 — Analytic STEP post-process A/B",
        "",
        f"- 时间: {meta.get('timestamp', '')}",
        f"- 基线 pred: `{meta.get('baseline_pred', '')}`",
        f"- analytic out: `{meta.get('analytic_pred', '')}`",
        f"- 评测 work: `{meta.get('work_dir', '')}`",
        f"- n_pred STEP: {meta.get('n_pred', '')}",
        "",
        "## 判定（赛题）",
        "",
        f"- Δsummary (analytic − raw) = **{dsum:+.6f}**",
        "- 若 Δ 明显为正：后处理可保留作推理默认；曲面类型问题仍建议后续进模型。",
        "- 若 Δ≈0：事后拟合几乎无用，优先做模型内 surf-type，而不是重训 P1。",
        "",
        "## 指标表",
        "",
        "| 指标 | raw (260725-002218) | analytic | Δ |",
        "|------|---------------------|----------|---|",
    ]
    keys = [
        ("summary", "summary"),
        ("gen_success_rate", "gen_success"),
        ("valid_ratio", "valid_ratio"),
        ("invalid_ratio", "IR≈invalid_ratio"),
        ("surface_f1", "surface_f1"),
        ("edge_f1", "edge_f1"),
        ("vertex_f1", "vertex_f1"),
        ("topo_f1", "topo_f1"),
        ("cd_surface", "CD_surface (median)"),
        ("cd_edge", "CD_edge"),
        ("cd_vertex", "CD_vertex"),
    ]
    for k, label in keys:
        a, b = raw.get(k), analytic.get(k)
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            delta = f"{float(b) - float(a):+.6f}"
        else:
            delta = "—"
        lines.append(f"| {label} | {_fmt(a)} | {_fmt(b)} | {delta} |")
    lines.extend(
        [
            "",
            "## 下一步（硬门禁）",
            "",
            "- 通过 → 继续 **R1**：现有 3view ckpt (`260723-162838`) + `postprocess_analytic=1` 全量 GT test。",
            "- 停在本报告：不自动开 P0 50ep / P1 全量训。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[R0] wrote report → {path}", flush=True)


def main() -> int:
    p = argparse.ArgumentParser(description="ECCV R0 analytic postprocess A/B gate")
    p.add_argument("--baseline-official", type=str, default=str(BASELINE_OFFICIAL))
    p.add_argument("--baseline-test-json", type=str, default=str(BASELINE_TEST_JSON))
    p.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT))
    p.add_argument("--report-dir", type=str, default=str(DEFAULT_REPORT_DIR))
    p.add_argument("--eval-py", type=str, default=str(DEFAULT_EVAL_PY))
    p.add_argument("--analytic-tol", type=float, default=1e-3)
    p.add_argument("--sew-tolerance", type=float, default=0.005)
    p.add_argument("--limit", type=int, default=0, help="debug: only first N STEPs")
    p.add_argument("--skip-postprocess", action="store_true")
    p.add_argument("--skip-eval", action="store_true")
    args = p.parse_args()

    baseline = Path(args.baseline_official)
    gt_src = baseline / "gt"
    pred_src = baseline / "pred"
    if not gt_src.is_dir() or not pred_src.is_dir():
        print(f"[R0] missing gt/pred under {baseline}", file=sys.stderr)
        return 2

    out_root = Path(args.out_dir)
    analytic_pred = out_root / "pred_analytic"
    work = out_root / "official_test_analytic"
    analytic_pred.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)

    steps = sorted(pred_src.glob("*.step"))
    if args.limit > 0:
        steps = steps[: int(args.limit)]
    print(f"[R0] n_pred={len(steps)} baseline={baseline}", flush=True)

    if not args.skip_postprocess:
        report = []
        for i, sp in enumerate(steps):
            dst = analytic_pred / sp.name
            try:
                info = postprocess_step_file(
                    sp,
                    dst,
                    analytic=True,
                    analytic_tol=float(args.analytic_tol),
                    sew_tolerance=float(args.sew_tolerance),
                )
            except Exception as exc:  # noqa: BLE001
                info = {"src": str(sp), "ok": False, "error": str(exc)}
            if not info.get("ok"):
                # Keep full eval coverage: fall back to raw STEP on failure.
                try:
                    shutil.copy2(sp, dst)
                    info["fallback_raw"] = True
                    info["ok_for_eval"] = True
                except OSError as copy_exc:
                    info["fallback_raw"] = False
                    info["ok_for_eval"] = False
                    info["copy_error"] = str(copy_exc)
            else:
                info["ok_for_eval"] = True
                info["fallback_raw"] = False
            report.append(info)
            if (i + 1) % 20 == 0 or i + 1 == len(steps):
                print(
                    f"[R0] postprocess {i + 1}/{len(steps)} "
                    f"last_ok={info.get('ok')} fallback={info.get('fallback_raw')}",
                    flush=True,
                )
        (analytic_pred / "postprocess_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        n_ok = sum(1 for r in report if r.get("ok"))
        n_fb = sum(1 for r in report if r.get("fallback_raw"))
        print(
            f"[R0] postprocess ok={n_ok}/{len(report)} fallback_raw={n_fb} → {analytic_pred}",
            flush=True,
        )

    # Prepare eval workdir: copy gt + analytic pred (eval.py expects cwd/gt, cwd/pred)
    work_gt = work / "gt"
    work_pred = work / "pred"
    if work_gt.exists():
        shutil.rmtree(work_gt)
    if work_pred.exists():
        shutil.rmtree(work_pred)
    shutil.copytree(gt_src, work_gt)
    work_pred.mkdir(parents=True)
    for sp in steps:
        src = analytic_pred / sp.name
        if src.is_file():
            shutil.copy2(src, work_pred / sp.name)

    raw_base: dict[str, Any] = {}
    tj = Path(args.baseline_test_json)
    if tj.is_file():
        blob = json.loads(tj.read_text(encoding="utf-8"))
        off = dict(blob.get("official") or {})
        # fill CD if missing by reusing blob top-level absences
        raw_base = _row_from_official(
            off,
            gen_n=int(blob.get("n_generated") or 0) or None,
            n_req=int(blob.get("n_requested") or 0) or None,
        )
        for k in ("summary", "surface_f1", "edge_f1", "vertex_f1", "topo_f1", "valid_ratio"):
            if k in blob and blob[k] is not None:
                raw_base[k] = float(blob[k])
        if blob.get("gen_success_rate") is not None:
            raw_base["gen_success_rate"] = float(blob["gen_success_rate"])

    analytic_metrics: dict[str, Any] = {}
    if not args.skip_eval:
        print(f"[R0] running official eval cwd={work}", flush=True)
        analytic_metrics = run_official_eval(work, eval_py=Path(args.eval_py), timeout_sec=7200)
        (work / "metrics_analytic.json").write_text(
            json.dumps(analytic_metrics, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(
            f"[R0] analytic summary={analytic_metrics.get('summary')} "
            f"surface_f1={analytic_metrics.get('surface_f1')} "
            f"cd_surface={analytic_metrics.get('cd_surface')}",
            flush=True,
        )

    n_gen = len(list(work_pred.glob("*.step")))
    analytic_row = _row_from_official(
        analytic_metrics,
        gen_n=n_gen,
        n_req=int(raw_base.get("n_requested") or 348),
    )

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    metrics_out = {
        "gate": "R0",
        "timestamp": stamp,
        "baseline_official": str(baseline),
        "raw": raw_base,
        "analytic": analytic_row,
        "analytic_official_raw": {
            k: v for k, v in analytic_metrics.items() if not str(k).startswith("_")
        },
    }
    (out_root / "R0_metrics.json").write_text(
        json.dumps(metrics_out, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    report_path = Path(args.report_dir) / f"R0_postprocess_ab_{stamp}.md"
    latest = Path(args.report_dir) / "R0_latest.md"
    meta = {
        "timestamp": stamp,
        "baseline_pred": str(pred_src),
        "analytic_pred": str(analytic_pred),
        "work_dir": str(work),
        "n_pred": n_gen,
    }
    _write_report(report_path, raw=raw_base, analytic=analytic_row, meta=meta)
    shutil.copy2(report_path, latest)

    print(
        f"[R0] DONE raw.summary={raw_base.get('summary')} "
        f"analytic.summary={analytic_row.get('summary')} "
        f"Δ={float(analytic_row.get('summary') or 0) - float(raw_base.get('summary') or 0):+.6f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
