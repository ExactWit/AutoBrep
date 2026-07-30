#!/usr/bin/env python3
"""Summarize STEP gen failure taxonomy from official_* / test metrics.json gen_log."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--metrics-json", required=True, help="metrics.json with gen_log")
    p.add_argument("--out", default="", help="optional markdown report path")
    args = p.parse_args()
    d = json.loads(Path(args.metrics_json).read_text(encoding="utf-8"))
    log = d.get("gen_log") or []
    n = len(log)
    n_ok = sum(1 for g in log if g.get("ok"))
    errs: Counter[str] = Counter()
    for g in log:
        if g.get("ok"):
            continue
        e = str(g.get("error") or "unknown")
        # collapse exception detail after first colon for taxonomy
        if e.startswith("exception:"):
            parts = e.split(":", 2)
            e = ":".join(parts[:2]) if len(parts) >= 2 else e
        errs[e] += 1
    lines = [
        f"# Gen failure taxonomy",
        "",
        f"- source: `{args.metrics_json}`",
        f"- n={n} ok={n_ok} fail={n - n_ok} gen_success={n_ok / max(n, 1):.4f}",
        "",
        "| error | count | share_of_fail |",
        "|-------|------:|-------------:|",
    ]
    denom = max(n - n_ok, 1)
    for e, c in errs.most_common():
        lines.append(f"| `{e}` | {c} | {c / denom:.1%} |")
    lines += [
        "",
        "## Priority notes",
        "",
        "- `decode_failed`: AR token 无法解析为面/边几何（序列非法 / reshape 失败）。",
        "- `rebuild_failed`: decode 成功但 OCC 重建/缝合失败（当前主因）。",
        "- 目标顺序：先抬 gen_success（降 decode/rebuild 失败），再抬 official summary。",
        "",
    ]
    text = "\n".join(lines)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
