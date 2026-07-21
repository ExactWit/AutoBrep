#!/usr/bin/env python3
"""ECCV public-train STEP → AutoBrep parquet under processed/autobrep/."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


def resolve_eccv_dataset_root(data_dir: str | Path) -> Path:
    """
    Launcher often passes ``.../processed`` or ``.../processed/autobrep``.
    Return the challenge dataset root that contains ``train/target_step``.
    """
    root = Path(data_dir).resolve()
    candidates = [
        root,
        root.parent,
        root.parent.parent,
    ]
    # Also try sibling/parent when under processed/
    if root.name == "autobrep":
        candidates.append(root.parents[1])
    if root.name == "processed":
        candidates.append(root.parent)
    for c in candidates:
        if (c / "train" / "target_step").is_dir():
            return c
    raise FileNotFoundError(
        f"Cannot find train/target_step under {root} (or parents). "
        "Pass ECCV challenge root or its processed/ directory."
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Preprocess ECCV STEP → AutoBrep parquet")
    p.add_argument(
        "--data-dir",
        default="/data/hdd/datasets/eccv2026ws-cad-data",
        help="ECCV dataset root",
    )
    p.add_argument(
        "--datasplit",
        default="",
        help="Path to datasplit.json (default: {data-dir}/processed/datasplit.json)",
    )
    p.add_argument("--max-face", type=int, default=200)
    p.add_argument("--max-edge", type=int, default=1000)
    p.add_argument("--shard-size", type=int, default=100)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument(
        "--limit-samples",
        type=int,
        default=0,
        help="If >0, only process this many sample ids (debug)",
    )
    p.add_argument(
        "--out-dir",
        default="",
        help="Override output root (default: {data-dir}/processed/autobrep)",
    )
    return p.parse_args()


def _process_one(args: tuple[str, str, int, int]) -> dict[str, Any]:
    """Worker: (sample_id, step_path, max_face, max_edge) → status dict."""
    sample_id, step_path, max_face, max_edge = args
    # Local imports for spawn-safe workers
    sys.path.insert(
        0, str(Path(__file__).resolve().parents[1] / "core" / "src")
    )
    from autobrep.data.step_to_autobrep import (
        StepExtractError,
        extract_autobrep_from_step,
        result_to_row,
    )

    try:
        result = extract_autobrep_from_step(
            step_path, max_face=max_face, max_edge=max_edge
        )
        row = result_to_row(sample_id, result)
        return {"ok": True, "sample_id": sample_id, "row": row}
    except StepExtractError as e:
        return {"ok": False, "sample_id": sample_id, "error": str(e)}
    except Exception as e:  # noqa: BLE001
        return {
            "ok": False,
            "sample_id": sample_id,
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
        }


def _write_shard(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path)


def main() -> int:
    args = parse_args()
    data_dir = resolve_eccv_dataset_root(args.data_dir)
    step_dir = data_dir / "train" / "target_step"
    if not step_dir.is_dir():
        print(f"[preprocess] ERROR: missing {step_dir}", file=sys.stderr)
        return 1

    split_path = (
        Path(args.datasplit)
        if args.datasplit
        else data_dir / "processed" / "datasplit.json"
    )
    if not split_path.is_file():
        print(f"[preprocess] ERROR: missing datasplit {split_path}", file=sys.stderr)
        return 1

    split_data = json.loads(split_path.read_text(encoding="utf-8"))
    splits: dict[str, list[str]] = {
        k: list(v) for k, v in split_data["splits"].items() if k != "public_test"
    }
    # Map sample_id → split (train/val/test only; all STEP live under train/)
    id_to_split: dict[str, str] = {}
    for split_name, ids in splits.items():
        for sid in ids:
            id_to_split[str(sid)] = split_name

    all_steps = sorted(step_dir.glob("*.step")) + sorted(step_dir.glob("*.stp"))
    jobs: list[tuple[str, str, int, int]] = []
    for step_path in all_steps:
        sid = step_path.stem
        if sid not in id_to_split:
            # Still process; put in train if unknown (should be rare)
            id_to_split[sid] = "train"
        jobs.append((sid, str(step_path), args.max_face, args.max_edge))

    if args.limit_samples > 0:
        jobs = jobs[: args.limit_samples]

    out_root = (
        Path(args.out_dir).resolve()
        if args.out_dir
        else data_dir / "processed" / "autobrep"
    )
    for split_name in ("train", "val", "test"):
        (out_root / split_name).mkdir(parents=True, exist_ok=True)

    # Soft-link / copy datasplit for convenience
    dst_split = out_root / "datasplit.json"
    if not dst_split.exists():
        try:
            dst_split.symlink_to(split_path.resolve())
        except OSError:
            shutil.copy2(split_path, dst_split)

    print(
        f"[preprocess] steps={len(jobs)} workers={args.num_workers} → {out_root}",
        flush=True,
    )

    buckets: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "val": [],
        "test": [],
    }
    shard_idx = {"train": 0, "val": 0, "test": 0}
    ok_n = fail_n = 0
    fails: list[dict[str, str]] = []

    def flush_if_needed(split_name: str, force: bool = False) -> None:
        buf = buckets[split_name]
        while len(buf) >= args.shard_size or (force and buf):
            take = buf[: args.shard_size]
            del buf[: args.shard_size]
            out_path = (
                out_root
                / split_name
                / f"autobrep-{split_name}-{shard_idx[split_name]:05d}.parquet"
            )
            _write_shard(out_path, take)
            print(
                f"[preprocess] wrote {out_path.name} rows={len(take)}",
                flush=True,
            )
            shard_idx[split_name] += 1

    workers = max(1, int(args.num_workers))
    if workers == 1:
        results_iter = (_process_one(j) for j in jobs)
        for res in results_iter:
            sid = res["sample_id"]
            split_name = id_to_split.get(sid, "train")
            if res["ok"]:
                buckets[split_name].append(res["row"])
                ok_n += 1
                flush_if_needed(split_name)
            else:
                fail_n += 1
                fails.append({"sample_id": sid, "error": res.get("error", "")})
            if (ok_n + fail_n) % 50 == 0:
                print(
                    f"[preprocess] progress {ok_n + fail_n}/{len(jobs)} "
                    f"ok={ok_n} fail={fail_n}",
                    flush=True,
                )
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_process_one, j): j[0] for j in jobs}
            done = 0
            for fut in as_completed(futs):
                res = fut.result()
                sid = res["sample_id"]
                split_name = id_to_split.get(sid, "train")
                if res["ok"]:
                    buckets[split_name].append(res["row"])
                    ok_n += 1
                    flush_if_needed(split_name)
                else:
                    fail_n += 1
                    fails.append({"sample_id": sid, "error": res.get("error", "")})
                done += 1
                if done % 50 == 0:
                    print(
                        f"[preprocess] progress {done}/{len(jobs)} "
                        f"ok={ok_n} fail={fail_n}",
                        flush=True,
                    )

    for split_name in ("train", "val", "test"):
        flush_if_needed(split_name, force=True)

    meta = {
        "data_dir": str(data_dir),
        "out_root": str(out_root),
        "max_face": args.max_face,
        "max_edge": args.max_edge,
        "jobs": len(jobs),
        "ok": ok_n,
        "fail": fail_n,
        "shards": dict(shard_idx),
        "views": [
            "transparent_shaded_edges",
            "hlg",
            "hlg_translucent",
            "techdraw_svg",
        ],
    }
    (out_root / "preprocess_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_root / "preprocess_fails.json").write_text(
        json.dumps(fails, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[preprocess] done ok={ok_n} fail={fail_n} → {out_root}", flush=True)
    return 0 if ok_n > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
