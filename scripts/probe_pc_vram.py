#!/usr/bin/env python3
"""Smoke / VRAM probe for frozen-backbone PC conditioning on one GPU."""

from __future__ import annotations

import argparse
import json

import torch


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--weight-folder", default="/data/hdd/outputs/AutoBrep")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--seq-len", type=int, default=512, help="padded token length for probe")
    p.add_argument("--pc-points", type=int, default=2048)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    from pathlib import Path
    import sys

    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo / "core" / "src"))

    from autobrep.models.autoregressive import AutoBrepPCModel

    w = Path(args.weight_folder)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    model = AutoBrepPCModel(
        surf_fsq_ckpt=str(w / "surf-fsq.ckpt"),
        edge_fsq_ckpt=str(w / "edge-fsq.ckpt"),
        ar_ckpt=str(w / "ar.ckpt"),
        freeze_backbone=True,
        max_seq=args.seq_len,
        max_face=200,
        bit=10,
        surf_codebook_size=1024,
        edge_codebook_size=1024,
        depth=16,
        heads=32,
        dim=2048,
        kv_groups=8,
        pc_num_latents=64,
    )
    model.to(device)
    model.train()

    b = args.batch_size
    # Minimal fake batch: random tokens in vocab range, random UV grids, PC
    vocab = model.face_z_pad + 1024 + 1024
    seq = torch.randint(0, min(vocab, 100), (b, args.seq_len), device=device)
    seq[:, 0] = 0  # BOS
    face_ncs = torch.randn(b, 8, 32, 32, 3, device=device, dtype=torch.bfloat16)
    edge_ncs = torch.randn(b, 16, 32, 3, device=device, dtype=torch.bfloat16)
    # Mark a few face/edge z slots so copy_fsq_code path is exercised lightly
    # Use only control-ish tokens to avoid index errors in copy_fsq_code
    seq[:] = -1
    seq[:, 0] = 0
    seq[:, 1] = 1  # EOS-ish stop early content
    pc = torch.randn(b, args.pc_points, 3, device=device)

    # Direct path: prepend + CE on short seq without FSQ encode of huge grids
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        prepend = model.encode_point_cloud(pc)
        # Build a short valid-ish token row for CE
        tokens = torch.randint(0, 50, (b, args.seq_len), device=device)
        loss = model.cad_gpt(tokens, prepend_embeds=prepend)

    loss.backward()

    trainable = sum(p.numel() for p in model.trainable_params)
    total = sum(p.numel() for p in model.parameters())
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)

    out = {
        "device": str(device),
        "batch_size": b,
        "seq_len": args.seq_len,
        "pc_points": args.pc_points,
        "params_total_M": round(total / 1e6, 2),
        "params_trainable_M": round(trainable / 1e6, 2),
        "params_frozen_M": round(frozen / 1e6, 2),
        "loss": float(loss.detach().cpu()),
    }
    if device.type == "cuda":
        out["peak_allocated_GB"] = round(torch.cuda.max_memory_allocated() / 1e9, 3)
        out["peak_reserved_GB"] = round(torch.cuda.max_memory_reserved() / 1e9, 3)
        out["gpu_name"] = torch.cuda.get_device_name(0)
        out["gpu_total_GB"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
