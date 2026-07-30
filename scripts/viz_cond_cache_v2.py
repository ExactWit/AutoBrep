#!/usr/bin/env python3
"""Visualize cond_cache_v2 inputs for all samples (train/val/test/public_test).

Layout per PNG (same as Stage A inspection panels):
  TOP:    3 render images (CNN)
  MID:    TechDraw sheet denormalized from prim tensors + L-layout boxes
  BOTTOM: 3 views in network unit space [-1, 1] (PrimEncoder input)

train/val/test: load ``processed/cond_cache_v2/{split}/{id}.pt``
public_test:    build the same tensors online from DXF/SVG + renders
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Arc, Circle, Rectangle

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "core" / "src"))

from autobrep.data.eccv_data import (  # noqa: E402
    load_render_views,
    load_techdraw_geometry,
)

VIEW_LABELS = ("view0", "view1", "view2")
RENDER_LABELS = ("hlg", "hlg_translucent", "transparent")
TYPE_NAMES = {
    0: "line",
    1: "arc",
    2: "circle",
    3: "spline",
    4: "ellipse",
    5: "lwpoly",
    6: "other",
}
TYPE_COLORS = {
    0: "#0077BB",
    1: "#EE7733",
    2: "#009988",
    3: "#CC3311",
    4: "#33BBEE",
    5: "#EE3377",
    6: "#888888",
}
LT_STYLE = {0: "-", 1: "--", 2: "-.", 3: ":"}
VIEW_EDGE = ("#1f77b4", "#ff7f0e", "#2ca02c")

PUBLIC_RENDER = {
    "render_hlg": "test_public/render_3d/hlg_perspective/{sid}.png",
    "render_hlg_translucent": (
        "test_public/render_3d/hlg_translucent_faces_perspective/{sid}.png"
    ),
    "render_transparent": (
        "test_public/render_3d/transparent_shaded_edges_perspective/{sid}.png"
    ),
}


def _denorm_xy(xy: np.ndarray, bmin: np.ndarray, span: np.ndarray) -> np.ndarray:
    u = (np.asarray(xy, dtype=np.float64) + 1.0) * 0.5
    return u * span + bmin


def _denorm_r(r_norm: float, scale: float) -> float:
    return float(r_norm) * 0.5 * scale


def _draw_prim_sheet(
    ax,
    tid: int,
    geom,
    lt: int,
    bmin: np.ndarray,
    span: np.ndarray,
    scale: float,
    color: str,
    lw: float = 0.9,
) -> None:
    g = np.asarray(geom, dtype=np.float64)
    ls = LT_STYLE.get(int(lt), "-")
    if tid == 0:
        p0 = _denorm_xy(g[0:2], bmin, span)
        p1 = _denorm_xy(g[2:4], bmin, span)
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=color, lw=lw, ls=ls)
    elif tid in (1, 2):
        c = _denorm_xy(g[0:2], bmin, span)
        r = _denorm_r(g[2], scale)
        if r <= 0:
            return
        if tid == 2:
            ax.add_patch(
                Circle((c[0], c[1]), r, fill=False, edgecolor=color, lw=lw, ls=ls)
            )
        else:
            a0 = math.degrees(float(g[3]))
            a1 = math.degrees(float(g[4]))
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
    elif tid == 4:
        c = _denorm_xy(g[0:2], bmin, span)
        maj = np.asarray(g[2:4], dtype=float) * scale
        a = float(np.linalg.norm(maj))
        ratio = float(g[4]) if abs(g[4]) > 1e-8 else 0.5
        if a < 1e-8:
            return
        u = maj / a
        v = np.array([-u[1], u[0]]) * (a * ratio)
        ts = np.linspace(0, 2 * math.pi, 64)
        pts = c + np.outer(np.cos(ts), u * a) + np.outer(np.sin(ts), v)
        ax.plot(pts[:, 0], pts[:, 1], color=color, lw=lw, ls=ls)
    else:
        pts = np.stack([_denorm_xy(g[i : i + 2], bmin, span) for i in range(0, 8, 2)])
        ax.plot(pts[:, 0], pts[:, 1], color=color, lw=lw, ls=ls)


def _draw_prim_unit(ax, tid: int, geom, lt: int, color: str, lw: float = 1.0) -> None:
    g = np.asarray(geom, dtype=np.float64)
    ls = LT_STYLE.get(int(lt), "-")
    if tid == 0:
        ax.plot([g[0], g[2]], [g[1], g[3]], color=color, lw=lw, ls=ls)
    elif tid in (1, 2):
        r_plot = float(g[2])
        if r_plot <= 0:
            return
        if tid == 2:
            ax.add_patch(
                Circle((g[0], g[1]), r_plot, fill=False, edgecolor=color, lw=lw, ls=ls)
            )
        else:
            a0 = math.degrees(float(g[3]))
            a1 = math.degrees(float(g[4]))
            ax.add_patch(
                Arc(
                    (g[0], g[1]),
                    2 * r_plot,
                    2 * r_plot,
                    angle=0,
                    theta1=a0,
                    theta2=a1,
                    color=color,
                    lw=lw,
                    ls=ls,
                )
            )
    else:
        pts = np.array([[g[i], g[i + 1]] for i in range(0, 8, 2)])
        ax.plot(pts[:, 0], pts[:, 1], color=color, lw=lw, ls=ls)


def _load_from_cache(cache_root: Path, split: str, sid: str) -> dict[str, Any] | None:
    path = cache_root / split / f"{sid}.pt"
    if not path.is_file():
        return None
    return torch.load(path, map_location="cpu", weights_only=False)


def _public_row(sid: str) -> dict[str, str]:
    return {
        "sample_id": sid,
        "techdraw_dxf_path": f"test_public/techdraw/dxf/{sid}.dxf",
        "techdraw_svg_path": f"test_public/techdraw/svg/{sid}.svg",
        **{k: v.format(sid=sid) for k, v in PUBLIC_RENDER.items()},
    }


def _load_public(dataset_root: Path, sid: str) -> dict[str, Any]:
    row = _public_row(sid)
    images = torch.from_numpy(load_render_views(dataset_root, row)).float()
    dxf = load_techdraw_geometry(dataset_root, row)
    return {
        "sample_id": sid,
        "images": images,
        **dxf,
        "meta": {"split_algo": "l_layout", "source": "online_public_test"},
    }


def render_one(
    payload: dict[str, Any],
) -> dict[str, Any]:
    split = payload["split"]
    sid = payload["sample_id"]
    out_png = Path(payload["out_png"])
    try:
        if out_png.is_file() and not payload.get("overwrite"):
            return {"sample_id": sid, "split": split, "ok": True, "skipped": True}

        if split == "public_test":
            d = _load_public(Path(payload["dataset_root"]), sid)
        else:
            d = _load_from_cache(Path(payload["cache_root"]), split, sid)
            if d is None:
                return {
                    "sample_id": sid,
                    "split": split,
                    "ok": False,
                    "error": "missing_cache",
                }

        images = d["images"]
        fig = plt.figure(figsize=(14, 10), dpi=int(payload.get("dpi", 110)))

        for i in range(3):
            ax = fig.add_axes([0.04 + i * 0.31, 0.70, 0.28, 0.26])
            ax.imshow(images[i].permute(1, 2, 0).numpy().clip(0, 1))
            ax.set_title(f"render[{i}] {RENDER_LABELS[i]}  (CNN)", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])

        ax_s = fig.add_axes([0.04, 0.38, 0.58, 0.28])
        all_min = np.array([1e9, 1e9], dtype=np.float64)
        all_max = np.array([-1e9, -1e9], dtype=np.float64)
        view_stats: list[str] = []
        for v in range(3):
            mask = d["prim_mask"][v]
            types = d["prim_types"][v]
            lts = d["prim_linetypes"][v]
            geom = d["prim_geom"][v]
            bmin = d["dxf_bbox_min"][v].numpy().astype(np.float64)
            bmax = d["dxf_bbox_max"][v].numpy().astype(np.float64)
            span = np.maximum(bmax - bmin, 1e-6)
            scale = float(np.max(span))
            all_min = np.minimum(all_min, bmin)
            all_max = np.maximum(all_max, bmax)
            n = int(mask.sum())
            cnt = Counter(
                TYPE_NAMES[int(types[j])]
                for j in range(mask.shape[0])
                if bool(mask[j])
            )
            view_stats.append(f"{VIEW_LABELS[v]}: n={n} {dict(cnt)}")
            for j in range(mask.shape[0]):
                if not bool(mask[j]):
                    continue
                tid = int(types[j])
                _draw_prim_sheet(
                    ax_s,
                    tid,
                    geom[j],
                    int(lts[j]),
                    bmin,
                    span,
                    scale,
                    TYPE_COLORS[tid],
                    lw=0.85,
                )
            ax_s.add_patch(
                Rectangle(
                    bmin,
                    span[0],
                    span[1],
                    fill=False,
                    ls="--",
                    edgecolor=VIEW_EDGE[v],
                    lw=1.4,
                    alpha=0.85,
                )
            )
            ax_s.text(
                bmin[0],
                bmax[1],
                f" {VIEW_LABELS[v]} n={n}",
                color=VIEW_EDGE[v],
                fontsize=8,
                va="bottom",
            )

        if np.isfinite(all_min).all() and (all_max > all_min).all():
            pad = (all_max - all_min) * 0.05
            ax_s.set_xlim(all_min[0] - pad[0], all_max[0] + pad[0])
            ax_s.set_ylim(all_min[1] - pad[1], all_max[1] + pad[1])
        ax_s.set_aspect("equal", adjustable="box")
        ax_s.invert_yaxis()
        ax_s.grid(True, alpha=0.15)
        ax_s.set_title(
            "TechDraw sheet (denorm from prim tensors) + L-layout boxes",
            fontsize=9,
        )

        ax_l = fig.add_axes([0.66, 0.38, 0.30, 0.28])
        ax_l.axis("off")
        meta = d.get("meta") or {}
        lines = [
            "Model inputs:",
            "  images → CNN (top)",
            "  prim_* → PrimEncoder (bottom)",
            "  geom local-normalized [-1,1]",
            f"split_algo={meta.get('split_algo', '?')}",
            f"sample={split}/{sid}",
            "",
            *view_stats,
        ]
        ax_l.text(0, 1, "\n".join(lines), va="top", family="monospace", fontsize=8)

        for v in range(3):
            ax = fig.add_axes([0.04 + v * 0.31, 0.05, 0.28, 0.28])
            mask = d["prim_mask"][v]
            types = d["prim_types"][v]
            lts = d["prim_linetypes"][v]
            geom = d["prim_geom"][v]
            n = int(mask.sum())
            for j in range(mask.shape[0]):
                if not bool(mask[j]):
                    continue
                tid = int(types[j])
                _draw_prim_unit(
                    ax, tid, geom[j], int(lts[j]), TYPE_COLORS[tid], lw=1.0
                )
            ax.set_xlim(-1.15, 1.15)
            ax.set_ylim(-1.15, 1.15)
            ax.set_aspect("equal")
            ax.invert_yaxis()
            ax.grid(True, alpha=0.25)
            ax.set_title(
                f"TD {VIEW_LABELS[v]}  n={n}  (unit [-1,1])",
                fontsize=9,
            )

        fig.suptitle(
            f"cond_cache_v2  {split}/{sid}  |  "
            "TOP=renders · MID=denorm sheet · BOTTOM=unit prims",
            fontsize=11,
            y=0.995,
        )
        handles = [
            plt.Line2D([0], [0], color=TYPE_COLORS[i], lw=2.5, label=TYPE_NAMES[i])
            for i in range(7)
        ]
        fig.legend(
            handles=handles,
            loc="lower center",
            ncol=7,
            fontsize=8,
            frameon=False,
            bbox_to_anchor=(0.5, -0.01),
        )
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, bbox_inches="tight")
        plt.close(fig)
        return {"sample_id": sid, "split": split, "ok": True, "skipped": False}
    except Exception as exc:  # noqa: BLE001
        plt.close("all")
        return {
            "sample_id": sid,
            "split": split,
            "ok": False,
            "error": f"{type(exc).__name__}:{exc}",
            "tb": traceback.format_exc()[-400:],
        }


def _list_ids_cache(cache_root: Path, split: str) -> list[str]:
    d = cache_root / split
    if not d.is_dir():
        return []
    return sorted(p.stem for p in d.glob("*.pt"))


def _list_ids_public(dataset_root: Path) -> list[str]:
    dxf_dir = dataset_root / "test_public" / "techdraw" / "dxf"
    if not dxf_dir.is_dir():
        return []
    return sorted(p.stem for p in dxf_dir.glob("*.dxf"))


def _write_index(out_root: Path, splits: list[str], counts: dict[str, int]) -> None:
    rows = []
    for sp in splits:
        n = counts.get(sp, 0)
        rows.append(
            f'<li><a href="{sp}/">{sp}</a> ({n} png)</li>'
        )
    html = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>cond_cache_v2 viz</title></head><body>"
        "<h1>cond_cache_v2 visualization</h1>"
        "<p>TOP=renders · MID=denorm TechDraw sheet · BOTTOM=unit prims</p>"
        f"<ul>{''.join(rows)}</ul>"
        "<p>Open a split folder and browse PNGs. "
        "Or use: <code>ls &lt;split&gt; | head</code></p>"
        "</body></html>"
    )
    (out_root / "index.html").write_text(html, encoding="utf-8")
    for sp in splits:
        sp_dir = out_root / sp
        if not sp_dir.is_dir():
            continue
        pngs = sorted(sp_dir.glob("*.png"))[:200]
        links = "".join(
            f'<div style="display:inline-block;margin:4px;text-align:center">'
            f'<a href="{p.name}"><img src="{p.name}" width="220" loading="lazy">'
            f"<br>{p.stem}</a></div>"
            for p in pngs
        )
        more = "" if len(list(sp_dir.glob("*.png"))) <= 200 else "<p>… truncated to 200</p>"
        (sp_dir / "index.html").write_text(
            f"<!doctype html><html><body><h2>{sp}</h2>{more}{links}</body></html>",
            encoding="utf-8",
        )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", default="/data/hdd/datasets/eccv2026ws-cad-data")
    p.add_argument(
        "--cache-root",
        default="/data/hdd/datasets/eccv2026ws-cad-data/processed/cond_cache_v2",
    )
    p.add_argument(
        "--out-root",
        default=(
            "/data/hdd/exps/runs/eccv2026ws-cad-data/gen/AutoBrep/"
            "stage_gates/cond_cache_v2_viz"
        ),
    )
    p.add_argument(
        "--splits",
        default="train,val,test,public_test",
    )
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--limit", type=int, default=0, help="per-split limit (0=all)")
    p.add_argument("--dpi", type=int, default=110)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    dataset_root = Path(args.dataset_root)
    cache_root = Path(args.cache_root)
    out_root = Path(args.out_root)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    out_root.mkdir(parents=True, exist_ok=True)

    payloads: list[dict[str, Any]] = []
    for split in splits:
        if split == "public_test":
            ids = _list_ids_public(dataset_root)
        else:
            ids = _list_ids_cache(cache_root, split)
        if args.limit > 0:
            ids = ids[: args.limit]
        for sid in ids:
            payloads.append(
                {
                    "split": split,
                    "sample_id": sid,
                    "dataset_root": str(dataset_root),
                    "cache_root": str(cache_root),
                    "out_png": str(out_root / split / f"{sid}.png"),
                    "dpi": args.dpi,
                    "overwrite": bool(args.overwrite),
                }
            )

    print(
        f"[viz_cond_cache_v2] n={len(payloads)} out={out_root} workers={args.workers}",
        flush=True,
    )
    results: list[dict[str, Any]] = []
    if args.workers <= 1:
        for i, pl in enumerate(payloads, 1):
            results.append(render_one(pl))
            if i % 50 == 0 or i == len(payloads):
                print(f"  progress {i}/{len(payloads)}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(render_one, pl): pl["sample_id"] for pl in payloads}
            for i, fut in enumerate(as_completed(futs), 1):
                results.append(fut.result())
                if i % 100 == 0 or i == len(futs):
                    n_ok = sum(1 for r in results if r.get("ok"))
                    n_fail = sum(1 for r in results if not r.get("ok"))
                    print(
                        f"  progress {i}/{len(futs)} ok={n_ok} fail={n_fail}",
                        flush=True,
                    )

    counts = {
        sp: sum(1 for r in results if r.get("split") == sp and r.get("ok"))
        for sp in splits
    }
    n_ok = sum(1 for r in results if r.get("ok"))
    n_skip = sum(1 for r in results if r.get("skipped"))
    n_fail = sum(1 for r in results if not r.get("ok"))
    failures = [r for r in results if not r.get("ok")]
    summary = {
        "out_root": str(out_root),
        "n": len(results),
        "n_ok": n_ok,
        "n_skip": n_skip,
        "n_fail": n_fail,
        "counts": counts,
        "failures": failures[:200],
    }
    (out_root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _write_index(out_root, splits, counts)
    (out_root / "README.txt").write_text(
        "cond_cache_v2 visualization\n"
        "TOP=3 renders (CNN) · MID=denorm TechDraw sheet · "
        "BOTTOM=unit [-1,1] prims (PrimEncoder)\n"
        "public_test built online (no .pt in cache).\n"
        f"summary: ok={n_ok} skip={n_skip} fail={n_fail}\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False)[:2000])
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
