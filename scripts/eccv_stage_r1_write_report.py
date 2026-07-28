#!/usr/bin/env python3
"""Write R1 stage report from a finished GT test metrics/test.json."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_REPORT_DIR = REPO / "docs" / "eccv_stage_reports"
BASELINE = {
    "run": "260725-002218",
    "summary": 0.045996,
    "gen_success_rate": 0.35344827586206895,
    "valid_ratio": 1.0,
    "surface_f1": 0.064516,
    "edge_f1": 0.02070937615185879,
    "vertex_f1": 0.016469127475843767,
    "topo_f1": 0.018589,
    "n_generated": 123,
    "n_requested": 348,
}
R0_DELTA = 0.000287  # analytic-only on old preds


def _fmt(v, digits=6):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--test-json", required=True)
    p.add_argument("--run-dir", default="")
    p.add_argument("--jid", default="")
    p.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    args = p.parse_args()

    blob = json.loads(Path(args.test_json).read_text(encoding="utf-8"))
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    keys = [
        ("summary", "summary"),
        ("gen_success_rate", "gen_success"),
        ("valid_ratio", "valid_ratio"),
        ("surface_f1", "surface_f1"),
        ("edge_f1", "edge_f1"),
        ("vertex_f1", "vertex_f1"),
        ("topo_f1", "topo_f1"),
        ("n_generated", "n_generated"),
    ]
    lines = [
        "# R1 — Existing 3view ckpt + analytic GT test",
        "",
        f"- 时间: {stamp}",
        f"- jid: `{args.jid or '—'}`",
        f"- run_dir: `{args.run_dir or Path(args.test_json).parents[1]}`",
        f"- checkpoint: `{blob.get('checkpoint', '')}`",
        f"- parent train: `260723-162838`",
        f"- postprocess_analytic: 1",
        f"- R0 参考: 旧 pred 事后 analytic Δsummary≈{R0_DELTA:+.6f}",
        "",
        "## 判定（硬门禁 1）",
        "",
        "- 对照基线 `260725-002218`（无 analytic / 旧 tip）。",
        "- 若 summary/gen 无明显提升：优先 **模型内 surf-type**，暂缓 P0 50ep / P1 全量。",
        "- 若 gen↑ 但 summary 平：后处理/合法率问题；若 summary↑：可考虑 hist-split 重训（R2）。",
        "- **默认停**：不自动 enqueue R2/P1。回复确认下一动作。",
        "",
        "## 指标表",
        "",
        "| 指标 | baseline 260725-002218 | R1 (analytic) | Δ |",
        "|------|------------------------|---------------|---|",
    ]
    for k, label in keys:
        a = BASELINE.get(k)
        b = blob.get(k)
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            delta = f"{float(b) - float(a):+.6f}"
        else:
            delta = "—"
        lines.append(f"| {label} | {_fmt(a)} | {_fmt(b)} | {delta} |")

    dsum = float(blob.get("summary") or 0) - float(BASELINE["summary"])
    lines.extend(
        [
            "",
            f"**Δsummary = {dsum:+.6f}**",
            "",
            "## 建议下一动作（勾选）",
            "",
            "- [ ] 开 R2：P0 hist-split 50ep 重训 + GT test",
            "- [ ] 直接做模型内曲面类型（surf-type token / builder）",
            "- [ ] 开 P1-A / P1-B（条件编码）",
            "- [ ] 仅保留 analytic 后处理，先交 public submission",
            "",
        ]
    )
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / f"R1_ckpt_gt_test_{stamp}.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    shutil.copy2(path, report_dir / "R1_latest.md")
    print(f"[R1] wrote {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
