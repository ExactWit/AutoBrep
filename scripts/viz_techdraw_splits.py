#!/usr/bin/env python3
"""Visualize TechDraw split exactly as the training network does.

Pipeline mirrors ``load_techdraw_geometry`` in ``eccv_data.py``:
  extract DXF + SVG → merge → filter_and_merge
  → split_into_views (XY-Cut regions → bbox-overlap assign)
  → assign_loop_groups per view.

Outputs per sample:
  - full sheet + 3 views colored by primitive type (7 classes)
  - view assignment shown as dashed bbox outlines
  - sidecar JSON with counts
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle, Rectangle
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "core" / "src"))

from autobrep.data.techdraw_dxf import (  # noqa: E402
    assign_loop_groups,
    extract_dxf_primitives,
    extract_svg_primitives,
    filter_and_merge,
    merge_dxfir,
    split_into_views,
)
from autobrep.data.techdraw_dxf.extract import _collect_points  # noqa: E402
from autobrep.data.techdraw_dxf.schema import (  # noqa: E402
    NUM_PRIM_TYPES,
    PRIM_TYPE_NAMES,
    DxfIR,
    PrimIR,
)

VIEW_NAMES = ("view0_TL", "view1_BL", "view2_TR")
VIEW_COLORS = ("#1f77b4", "#ff7f0e", "#2ca02c")  # blue / orange / green — view bbox only
# 7 prim types (NUM_PRIM_TYPES), high-contrast palette for sheet + panels
TYPE_COLORS = {
    "line": "#0077BB",       # blue
    "arc": "#EE7733",        # orange
    "circle": "#009988",     # teal
    "spline": "#CC3311",     # red
    "ellipse": "#33BBEE",    # cyan
    "lwpolyline": "#EE3377", # magenta
    "other": "#BBBBBB",      # gray
}
TYPE_ORDER = [PRIM_TYPE_NAMES[i] for i in range(NUM_PRIM_TYPES)]
ROLE_LS = {"outer": "-", "inner": "--", "isolated": ":"}


def process_sample_views(
    dataset_root: Path, sample_id: str, *, split: str
) -> tuple[DxfIR, list[DxfIR], dict[str, Any]]:
    """Same logic as ``load_techdraw_geometry`` but keep IRs for drawing."""
    root = Path(dataset_root)
    if split == "public_test":
        dxf = root / "test_public" / "techdraw" / "dxf" / f"{sample_id}.dxf"
        svg = root / "test_public" / "techdraw" / "svg" / f"{sample_id}.svg"
    else:
        dxf = root / "train" / "techdraw" / "dxf" / f"{sample_id}.dxf"
        svg = root / "train" / "techdraw" / "svg" / f"{sample_id}.svg"

    parts: list[DxfIR] = []
    meta: dict[str, Any] = {
        "sample_id": sample_id,
        "split": split,
        "dxf": str(dxf) if dxf.is_file() else "",
        "svg": str(svg) if svg.is_file() else "",
        "ok": False,
    }
    try:
        if dxf.is_file():
            parts.append(extract_dxf_primitives(dxf))
    except Exception as exc:  # noqa: BLE001
        meta["dxf_error"] = f"{type(exc).__name__}:{exc}"
    try:
        if svg.is_file():
            parts.append(extract_svg_primitives(svg))
    except Exception as exc:  # noqa: BLE001
        meta["svg_error"] = f"{type(exc).__name__}:{exc}"

    if not parts:
        meta["error"] = "no_dxf_svg"
        return (
            DxfIR(0, [], np.zeros(2, np.float32), np.ones(2, np.float32)),
            [],
            meta,
        )

    merged = filter_and_merge(merge_dxfir(parts))
    views = split_into_views(merged, use_histogram=True)
    grouped: list[DxfIR] = []
    for v in views:
        prims = assign_loop_groups(list(v.prims[: int(v.n_prims)]))
        grouped.append(
            DxfIR(
                n_prims=len(prims),
                prims=prims,
                bbox_min=v.bbox_min,
                bbox_max=v.bbox_max,
            )
        )

    type_counts = Counter()
    role_counts = Counter()
    for v in grouped:
        for p in v.prims[: int(v.n_prims)]:
            type_counts[p.type] += 1
            role_counts[p.group_role] += 1

    meta.update(
        {
            "ok": True,
            "n_merged": int(merged.n_prims),
            "n_per_view": [int(v.n_prims) for v in grouped],
            "type_counts": dict(type_counts),
            "role_counts": dict(role_counts),
            "n_groups": len(
                {
                    int(p.group_id)
                    for v in grouped
                    for p in v.prims[: int(v.n_prims)]
                    if int(p.group_id) > 0
                }
            ),
        }
    )
    return merged, grouped, meta


def _draw_prim(ax, prim: PrimIR, *, color: str, lw: float = 0.8, ls: str = "-") -> None:
    params = prim.params
    t = prim.type
    try:
        if t == "line":
            s, e = params["start"][:2], params["end"][:2]
            ax.plot([s[0], e[0]], [s[1], e[1]], color=color, lw=lw, ls=ls, solid_capstyle="round")
        elif t == "circle":
            c = params["center"]
            r = float(params["radius"])
            ax.add_patch(
                Circle((c[0], c[1]), r, fill=False, edgecolor=color, lw=lw, ls=ls)
            )
        elif t == "arc":
            c = params["center"]
            r = float(params["radius"])
            a0 = math.degrees(float(params.get("start_angle", 0.0)))
            a1 = math.degrees(float(params.get("end_angle", 2 * math.pi)))
            ax.add_patch(
                Arc(
                    (c[0], c[1]),
                    2 * r,
                    2 * r,
                    angle=0,
                    theta1=a0,
                    theta2=a1,
                    color=color,
                    lw=lw,
                    ls=ls,
                )
            )
        elif t == "lwpolyline":
            pts = np.asarray(params.get("points") or [], dtype=np.float32)
            if len(pts) >= 2:
                xs, ys = pts[:, 0], pts[:, 1]
                if params.get("closed") and len(pts) > 2:
                    xs = np.r_[xs, xs[0]]
                    ys = np.r_[ys, ys[0]]
                ax.plot(xs, ys, color=color, lw=lw, ls=ls)
        elif t == "spline":
            pts = np.asarray(params.get("control_points") or [], dtype=np.float32)
            if len(pts) >= 2:
                ax.plot(pts[:, 0], pts[:, 1], color=color, lw=lw, ls=ls)
        elif t == "ellipse":
            # approximate with polyline using major axis + ratio
            c = np.asarray(params["center"][:2], dtype=np.float64)
            maj = np.asarray(params["major_axis"][:2], dtype=np.float64)
            ratio = abs(float(params.get("ratio", 1.0)))
            a = np.linalg.norm(maj)
            if a < 1e-9:
                return
            u = maj / a
            v = np.array([-u[1], u[0]]) * (a * ratio)
            ts = np.linspace(0, 2 * math.pi, 64, endpoint=True)
            pts = c + np.outer(np.cos(ts), u * a) + np.outer(np.sin(ts), v)
            ax.plot(pts[:, 0], pts[:, 1], color=color, lw=lw, ls=ls)
        else:
            pts = np.asarray(_collect_points(prim), dtype=np.float32)
            if len(pts) >= 2:
                ax.plot(pts[:, 0], pts[:, 1], color=color, lw=lw, ls=ls)
    except Exception:  # noqa: BLE001
        return


def _set_equal_lim(ax, bbox_min, bbox_max, pad_ratio: float = 0.05) -> None:
    mn = np.asarray(bbox_min, dtype=np.float64)
    mx = np.asarray(bbox_max, dtype=np.float64)
    span = np.maximum(mx - mn, 1e-3)
    pad = span * pad_ratio
    ax.set_xlim(mn[0] - pad[0], mx[0] + pad[0])
    ax.set_ylim(mn[1] - pad[1], mx[1] + pad[1])
    ax.set_aspect("equal", adjustable="box")
    ax.invert_yaxis()  # TechDraw sheet often y-down-ish; keep paper feel
    ax.grid(True, alpha=0.15, lw=0.4)


def _type_color(prim: PrimIR) -> str:
    return TYPE_COLORS.get(prim.type, TYPE_COLORS["other"])


def _draw_view_bbox(ax, v: DxfIR, *, color: str, label: str) -> None:
    if int(v.n_prims) <= 0:
        return
    mn = np.asarray(v.bbox_min, dtype=np.float64)
    mx = np.asarray(v.bbox_max, dtype=np.float64)
    w, h = float(mx[0] - mn[0]), float(mx[1] - mn[1])
    if w < 1e-6 or h < 1e-6:
        return
    ax.add_patch(
        Rectangle(
            (mn[0], mn[1]),
            w,
            h,
            fill=False,
            edgecolor=color,
            lw=1.6,
            ls="--",
            alpha=0.85,
            zorder=5,
        )
    )
    ax.text(
        mn[0],
        mn[1] - 0.02 * max(h, 1.0),
        label,
        color=color,
        fontsize=7,
        ha="left",
        va="top",
        zorder=6,
    )


def render_figure(
    sample_id: str,
    merged: DxfIR,
    views: list[DxfIR],
    meta: dict[str, Any],
    out_png: Path,
) -> None:
    fig = plt.figure(figsize=(14, 10), dpi=120)
    # top: full sheet colored by primitive type; dashed boxes = view split
    ax0 = fig.add_axes([0.04, 0.52, 0.58, 0.44])
    ax0.set_title(
        f"{sample_id}  |  merged={meta.get('n_merged')}  "
        f"views={meta.get('n_per_view')}  groups={meta.get('n_groups')}  "
        f"| color = prim type ({NUM_PRIM_TYPES})",
        fontsize=11,
    )
    for v in views:
        for prim in v.prims[: int(v.n_prims)]:
            ls = ROLE_LS.get(prim.group_role, "-")
            _draw_prim(ax0, prim, color=_type_color(prim), lw=0.85, ls=ls)
    for vi, v in enumerate(views):
        _draw_view_bbox(ax0, v, color=VIEW_COLORS[vi], label=VIEW_NAMES[vi])
    if int(merged.n_prims) > 0:
        _set_equal_lim(ax0, merged.bbox_min, merged.bbox_max)
    ax0.legend(
        handles=[
            plt.Line2D([0], [0], color=TYPE_COLORS[t], lw=2.5, label=t)
            for t in TYPE_ORDER
        ],
        loc="upper right",
        fontsize=7,
        title="prim type",
        framealpha=0.9,
    )

    # legend / counts on the right
    axL = fig.add_axes([0.64, 0.52, 0.32, 0.44])
    axL.axis("off")
    tc = meta.get("type_counts") or {}
    lines = [
        f"Primitive types: {NUM_PRIM_TYPES}",
        "",
    ]
    for t in TYPE_ORDER:
        lines.append(f"  {t:11s}  n={int(tc.get(t, 0)):4d}  {TYPE_COLORS[t]}")
    lines.extend(
        [
            "",
            "Stroke color = prim type (all panels)",
            "Dashed bbox  = view assignment",
            f"  {VIEW_NAMES[0]} bbox = blue",
            f"  {VIEW_NAMES[1]} bbox = orange",
            f"  {VIEW_NAMES[2]} bbox = green",
            "",
            "Line style = group_role:",
            "  solid=outer, dashed=inner, dotted=isolated",
            "",
            "Role counts:",
        ]
    )
    for k, c in sorted((meta.get("role_counts") or {}).items()):
        lines.append(f"  {k}: {c}")
    lines.extend(
        [
            "",
            "Pipeline:",
            "  DXF+SVG → filter_and_merge",
            "  → XY-Cut view regions",
            "  → bbox-overlap assign",
            "  → assign_loop_groups",
        ]
    )
    axL.text(
        0.0,
        1.0,
        "\n".join(lines),
        va="top",
        ha="left",
        family="monospace",
        fontsize=8,
        transform=axL.transAxes,
    )

    # bottom: 3 views, still type-colored; panel border = view slot
    for vi, v in enumerate(views):
        ax = fig.add_axes([0.04 + vi * 0.32, 0.06, 0.30, 0.40])
        ax.set_title(
            f"{VIEW_NAMES[vi]}  n={int(v.n_prims)}",
            fontsize=10,
            color=VIEW_COLORS[vi],
        )
        for prim in v.prims[: int(v.n_prims)]:
            ls = ROLE_LS.get(prim.group_role, "-")
            _draw_prim(ax, prim, color=_type_color(prim), lw=1.0, ls=ls)
        if int(v.n_prims) > 0:
            _set_equal_lim(ax, v.bbox_min, v.bbox_max)
        else:
            ax.set_xticks([])
            ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(VIEW_COLORS[vi])
            spine.set_linewidth(1.5)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _worker(payload: dict[str, Any]) -> dict[str, Any]:
    sample_id = payload["sample_id"]
    split = payload["split"]
    root = Path(payload["dataset_root"])
    out_dir = Path(payload["out_dir"])
    try:
        merged, views, meta = process_sample_views(root, sample_id, split=split)
        png = out_dir / split / f"{sample_id}.png"
        if meta.get("ok"):
            render_figure(sample_id, merged, views, meta, png)
            meta["png"] = str(png)
        else:
            meta["png"] = ""
        # sidecar
        js = out_dir / split / f"{sample_id}.json"
        js.parent.mkdir(parents=True, exist_ok=True)
        js.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return meta
    except Exception as exc:  # noqa: BLE001
        return {
            "sample_id": sample_id,
            "split": split,
            "ok": False,
            "error": f"{type(exc).__name__}:{exc}",
            "traceback": traceback.format_exc()[-500:],
        }


def load_all_ids(dataset_root: Path) -> list[tuple[str, str]]:
    ds = json.loads((dataset_root / "processed" / "datasplit.json").read_text())
    out: list[tuple[str, str]] = []
    for split, ids in (ds.get("splits") or {}).items():
        for sid in ids:
            out.append((str(sid), str(split)))
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="Visualize TechDraw view splits (network pipeline)")
    p.add_argument(
        "--dataset-root",
        default="/data/hdd/datasets/eccv2026ws-cad-data",
    )
    p.add_argument(
        "--out-dir",
        default="/data/hdd/exps/runs/eccv2026ws-cad-data/gen/AutoBrep/stage_gates/techdraw_viz",
    )
    p.add_argument("--workers", type=int, default=max(1, min(16, (os.cpu_count() or 4))))
    p.add_argument("--limit", type=int, default=0, help="debug: only first N samples")
    p.add_argument("--splits", default="train,val,test,public_test")
    args = p.parse_args()

    root = Path(args.dataset_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    want = {s.strip() for s in args.splits.split(",") if s.strip()}
    items = [(sid, sp) for sid, sp in load_all_ids(root) if sp in want]
    items.sort(key=lambda x: (x[1], x[0]))
    if args.limit > 0:
        items = items[: int(args.limit)]

    print(
        f"[viz_td] n={len(items)} workers={args.workers} out={out_dir}",
        flush=True,
    )
    payloads = [
        {
            "sample_id": sid,
            "split": sp,
            "dataset_root": str(root),
            "out_dir": str(out_dir),
        }
        for sid, sp in items
    ]

    results: list[dict[str, Any]] = []
    ok = 0
    with ProcessPoolExecutor(max_workers=int(args.workers)) as ex:
        futs = {ex.submit(_worker, pl): pl["sample_id"] for pl in payloads}
        for i, fut in enumerate(as_completed(futs), 1):
            meta = fut.result()
            results.append(meta)
            if meta.get("ok"):
                ok += 1
            if i % 100 == 0 or i == len(futs):
                print(f"[viz_td] {i}/{len(futs)} ok={ok}", flush=True)

    summary = {
        "n": len(results),
        "n_ok": ok,
        "n_fail": len(results) - ok,
        "out_dir": str(out_dir),
        "pipeline": "extract→filter_and_merge→XY-Cut regions→bbox-assign→loop_groups",
        "view_order": list(VIEW_NAMES),
        "by_split": {},
    }
    by = {}
    for r in results:
        sp = r.get("split", "?")
        by.setdefault(sp, {"n": 0, "ok": 0})
        by[sp]["n"] += 1
        by[sp]["ok"] += int(bool(r.get("ok")))
    summary["by_split"] = by
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    # compact index of failures
    fails = [r for r in results if not r.get("ok")]
    (out_dir / "failures.json").write_text(
        json.dumps(fails, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    # simple HTML index (first 200 ok per split + counts)
    html_lines = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        "<title>TechDraw split viz</title>",
        "<style>body{font-family:sans-serif} img{max-width:480px;border:1px solid #ccc;margin:6px}",
        ".row{display:flex;flex-wrap:wrap}</style></head><body>",
        f"<h1>TechDraw view split (network pipeline)</h1>",
        f"<p>ok={ok}/{len(results)} → <code>{out_dir}</code></p>",
    ]
    for sp in sorted(by.keys()):
        html_lines.append(f"<h2>{sp} ({by[sp]['ok']}/{by[sp]['n']})</h2><div class='row'>")
        pngs = sorted((out_dir / sp).glob("*.png"))[:120]
        for png in pngs:
            html_lines.append(
                f"<div><div>{png.stem}</div><a href='{sp}/{png.name}'>"
                f"<img src='{sp}/{png.name}' loading='lazy'></a></div>"
            )
        if len(list((out_dir / sp).glob("*.png"))) > 120:
            html_lines.append(f"<p>… more under {sp}/</p>")
        html_lines.append("</div>")
    html_lines.append("</body></html>")
    (out_dir / "index.html").write_text("\n".join(html_lines), encoding="utf-8")
    print(f"[viz_td] DONE ok={ok}/{len(results)} index={out_dir / 'index.html'}", flush=True)
    return 0 if ok > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
