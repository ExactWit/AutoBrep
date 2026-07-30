#!/usr/bin/env python3
"""Build processed/cond_cache_v2 for MM Stage A training acceleration.

Per sample .pt contains:
  images (3,3,H,W), prim_* TechDraw tensors (L-layout), surf_type_ids, meta
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "core" / "src"))

from autobrep.data.eccv_data import (  # noqa: E402
    RENDER_COLS,
    VIEW_SIZE,
    load_render_views,
    load_surf_type_ids,
    load_techdraw_geometry,
)
from autobrep.data.surf_types import SURF_TYPE_MAX_FACES  # noqa: E402

CACHE_VERSION = 2


def _process_one(payload: dict) -> dict:
    dataset_root = Path(payload["dataset_root"])
    out_root = Path(payload["out_root"])
    split = payload["split"]
    sample_id = payload["sample_id"]
    row = payload["row"]
    out_path = out_root / split / f"{sample_id}.pt"
    if out_path.is_file() and not payload.get("overwrite"):
        return {"sample_id": sample_id, "split": split, "ok": True, "skipped": True}
    try:
        images = torch.from_numpy(
            load_render_views(dataset_root, row, size=VIEW_SIZE)
        ).float()
        dxf = load_techdraw_geometry(dataset_root, row)
        surf = load_surf_type_ids(dataset_root, sample_id, split_hint=split)
        obj = {
            "sample_id": sample_id,
            "images": images,
            **{k: v for k, v in dxf.items()},
            "surf_type_ids": surf,
            "meta": {
                "cache_version": CACHE_VERSION,
                "split_algo": "l_layout",
                "render_cols": list(RENDER_COLS),
                "view_size": VIEW_SIZE,
                "surf_type_max_faces": SURF_TYPE_MAX_FACES,
                "techdraw_dxf_path": row.get("techdraw_dxf_path", ""),
                "techdraw_svg_path": row.get("techdraw_svg_path", ""),
            },
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(obj, out_path)
        return {
            "sample_id": sample_id,
            "split": split,
            "ok": True,
            "skipped": False,
            "path": str(out_path),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "sample_id": sample_id,
            "split": split,
            "ok": False,
            "error": f"{type(exc).__name__}:{exc}",
            "tb": traceback.format_exc()[-500:],
        }


def _rows_from_parquet(parquet_root: Path, split: str) -> list[dict]:
    import pyarrow.parquet as pq

    split_dir = parquet_root / split
    if not split_dir.is_dir():
        return []
    rows: list[dict] = []
    for pf in sorted(split_dir.glob("*.parquet")):
        table = pq.read_table(
            pf,
            columns=[
                "sample_id",
                "stem",
                "render_transparent",
                "render_hlg",
                "render_hlg_translucent",
                "techdraw_svg_path",
                "techdraw_dxf_path",
            ],
        )
        d = table.to_pydict()
        n = len(d["sample_id"])
        for i in range(n):
            sid = str(d["sample_id"][i] or d["stem"][i] or "")
            if not sid:
                continue
            rows.append(
                {
                    "sample_id": sid,
                    "stem": str(d["stem"][i] or sid),
                    "render_transparent": d["render_transparent"][i],
                    "render_hlg": d["render_hlg"][i],
                    "render_hlg_translucent": d["render_hlg_translucent"][i],
                    "techdraw_svg_path": d["techdraw_svg_path"][i],
                    "techdraw_dxf_path": d["techdraw_dxf_path"][i],
                }
            )
    return rows


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dataset-root",
        default="/data/hdd/datasets/eccv2026ws-cad-data",
    )
    p.add_argument(
        "--parquet-root",
        default="",
        help="default: <dataset-root>/processed/autobrep",
    )
    p.add_argument(
        "--out-root",
        default="",
        help="default: <dataset-root>/processed/cond_cache_v2",
    )
    p.add_argument("--splits", default="train,val,test")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=0, help="per-split limit (debug)")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    dataset_root = Path(args.dataset_root)
    parquet_root = (
        Path(args.parquet_root)
        if args.parquet_root
        else dataset_root / "processed" / "autobrep"
    )
    out_root = (
        Path(args.out_root)
        if args.out_root
        else dataset_root / "processed" / "cond_cache_v2"
    )
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    out_root.mkdir(parents=True, exist_ok=True)

    payloads = []
    for split in splits:
        rows = _rows_from_parquet(parquet_root, split)
        if args.limit > 0:
            rows = rows[: args.limit]
        for row in rows:
            payloads.append(
                {
                    "dataset_root": str(dataset_root),
                    "out_root": str(out_root),
                    "split": split,
                    "sample_id": row["sample_id"],
                    "row": row,
                    "overwrite": bool(args.overwrite),
                }
            )

    print(f"[cond_cache_v2] n={len(payloads)} out={out_root} workers={args.workers}")
    results = []
    if args.workers <= 1:
        for pl in payloads:
            results.append(_process_one(pl))
            if len(results) % 50 == 0:
                print(f"  progress {len(results)}/{len(payloads)}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_process_one, pl): pl["sample_id"] for pl in payloads}
            for i, fut in enumerate(as_completed(futs), 1):
                results.append(fut.result())
                if i % 50 == 0 or i == len(futs):
                    print(f"  progress {i}/{len(futs)}", flush=True)

    n_ok = sum(1 for r in results if r.get("ok"))
    n_skip = sum(1 for r in results if r.get("skipped"))
    n_fail = sum(1 for r in results if not r.get("ok"))
    manifest = {
        "cache_version": CACHE_VERSION,
        "out_root": str(out_root),
        "n": len(results),
        "n_ok": n_ok,
        "n_skip": n_skip,
        "n_fail": n_fail,
        "render_cols": list(RENDER_COLS),
        "split_algo": "l_layout",
        "failures": [r for r in results if not r.get("ok")][:50],
    }
    (out_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({k: manifest[k] for k in ("n", "n_ok", "n_skip", "n_fail")}, indent=2))
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
