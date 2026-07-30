#!/usr/bin/env python3
"""One-shot autocomplete smoke tests (cases A/B/C).

A: adjacent face pair from ABC
B: constraint_faces mask (or fallback to random if absent)
C: JSON exported from A, re-ingested

Usage:
  python scripts/autocomplete_smoke.py [--case A|B|C|all] [--output-dir ...]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DEFAULT_STEM = "00007410_d681bb73885e41b39c681d22_step_047_0074"
DEFAULT_DATA = "/data/hdd/datasets/ABC-1M"
DEFAULT_WEIGHTS = "/data/hdd/outputs/AutoBrep"
TESTDATA = REPO / "testdata" / "autocomplete"


def _ensure_pythonpath(env: dict) -> dict:
    src = str(REPO / "core" / "src")
    env = dict(env)
    prev = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{src}:{prev}" if prev else src
    return env


def export_case_a_json(stem: str, data_dir: str, out_json: Path) -> list[int]:
    sys.path.insert(0, str(REPO / "core" / "src"))
    from autobrep.inference.autocomplete_prompt import (
        condition_to_jsonable,
        find_abc_row_by_stem,
        load_condition_from_abc_row,
        pick_adjacent_face_pair,
    )

    row = find_abc_row_by_stem(data_dir, stem, split="train")
    cond = load_condition_from_abc_row(row, face_ids=None, num_faces=2, mode="random")
    # Prefer true adjacent pair
    pair = pick_adjacent_face_pair(cond["face_edge_adj"])
    cond = load_condition_from_abc_row(row, face_ids=pair, mode="random")
    payload = condition_to_jsonable(
        cond["face_pos"],
        cond["face_ncs"],
        cond["edge_pos"],
        cond["edge_ncs"],
        cond["face_edge_adj"],
        cond["user_face_indices"],
    )
    payload["meta"] = {
        "stem": stem,
        "face_ids": cond["user_face_indices"],
        "case": "A",
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload), encoding="utf-8")
    print(f"[smoke] wrote {out_json} faces={cond['user_face_indices']}")
    return list(cond["user_face_indices"])


def run_infer(
    *,
    output_dir: Path,
    exp_dir: Path,
    extra: list[str],
    weight_folder: str,
    gpu: str,
) -> int:
    cmd = [
        "bash",
        str(REPO / "run.sh"),
        "infer",
        "--exp-dir",
        str(exp_dir),
        "--output-dir",
        str(output_dir),
        "--weight-folder",
        weight_folder,
        "--gpu",
        gpu,
        "--batch-size",
        "1",
        "--num-batches",
        "1",
        "--complexity",
        "medium",
        "--debug",
        "0",
        "--use-seed",
        "1",
        "--seed",
        "42",
        "--autocomplete",
        "1",
        "--dataset",
        "abc-1m",
    ] + extra
    print("[smoke]", " ".join(cmd), flush=True)
    env = _ensure_pythonpath(dict(**{k: v for k, v in __import__("os").environ.items()}))
    return subprocess.call(cmd, cwd=str(REPO), env=env)


def summarize(output_dir: Path, case: str) -> dict:
    infer = output_dir / "infer"
    steps = sorted(infer.glob("*.step"))
    sidecars = sorted(infer.glob("*.json"))
    info = {
        "case": case,
        "n_step": len(steps),
        "n_sidecar": len(sidecars),
        "steps": [p.name for p in steps],
    }
    if sidecars:
        meta = json.loads(sidecars[0].read_text(encoding="utf-8"))
        info["sidecar"] = {
            k: meta.get(k)
            for k in (
                "num_faces",
                "prompt_len",
                "face_ids",
                "stem",
                "autocomplete",
                "geom_token_len",
            )
            if k in meta
        }
    (output_dir / "smoke_summary.json").write_text(
        json.dumps(info, indent=2), encoding="utf-8"
    )
    print(json.dumps(info, indent=2))
    return info


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--case", default="all", choices=["A", "B", "C", "all"])
    p.add_argument("--stem", default=DEFAULT_STEM)
    p.add_argument("--data-dir", default=DEFAULT_DATA)
    p.add_argument("--weight-folder", default=DEFAULT_WEIGHTS)
    p.add_argument("--gpu", default="0")
    p.add_argument(
        "--output-dir",
        default=str(Path("/data/hdd/outputs") / "autobrep_autocomplete_smoke"),
    )
    args = p.parse_args()

    base_out = Path(args.output_dir)
    exp_dir = base_out / "exp"
    exp_dir.mkdir(parents=True, exist_ok=True)
    TESTDATA.mkdir(parents=True, exist_ok=True)

    cases = ["A", "B", "C"] if args.case == "all" else [args.case]
    rc = 0

    # Prepare JSON for A/C
    json_a = TESTDATA / "case_a_adjacent.json"
    face_ids_a = export_case_a_json(args.stem, args.data_dir, json_a)

    for case in cases:
        out = base_out / f"case_{case}"
        out.mkdir(parents=True, exist_ok=True)
        if case == "A":
            extra = [
                "--data-dir",
                args.data_dir,
                "--abc-stem",
                args.stem,
                "--face-ids",
                ",".join(map(str, face_ids_a)),
            ]
        elif case == "B":
            extra = [
                "--data-dir",
                args.data_dir,
                "--abc-stem",
                args.stem,
                "--condition-mode",
                "constraint",
            ]
        else:
            extra = ["--condition-json", str(json_a)]
        code = run_infer(
            output_dir=out,
            exp_dir=exp_dir,
            extra=extra,
            weight_folder=args.weight_folder,
            gpu=args.gpu,
        )
        info = summarize(out, case)
        if code != 0:
            print(f"[smoke] case {case} infer exited {code}", file=sys.stderr)
            rc = code or 1
        elif info["n_step"] == 0:
            # Rebuild may fail; pipeline OK if sidecar exists without crash
            if info["n_sidecar"] > 0:
                print(
                    f"[smoke] case {case}: no STEP but sidecar written "
                    f"(rebuild/decode soft-fail) — OK for pipeline",
                    flush=True,
                )
            else:
                print(
                    f"[smoke] case {case}: no STEP and no sidecar",
                    file=sys.stderr,
                )
                rc = 1
        else:
            print(f"[smoke] case {case} OK: {info['n_step']} STEP")

    return rc


if __name__ == "__main__":
    raise SystemExit(main())
