#!/usr/bin/env python3
"""Create a host run for classical baseline (no training). Writes sentinel ckpt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--exp-dir", required=True)
    args = p.parse_args()
    exp = Path(args.exp_dir)
    ckpt_dir = exp / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    # Empty sentinel so exp_launcher test can resolve parent checkpoint.
    (ckpt_dir / "best.ckpt").write_bytes(b"classical_wm_host\n")
    (ckpt_dir / "last.ckpt").write_bytes(b"classical_wm_host\n")
    meta = {
        "method": "wesley_markowsky_classical",
        "note": "host stub; no DL weights",
    }
    (exp / "metrics").mkdir(parents=True, exist_ok=True)
    (exp / "metrics" / "classical_host.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[classical_host] wrote sentinels under {ckpt_dir}", flush=True)


if __name__ == "__main__":
    main()
