#!/usr/bin/env python3
"""exp_launcher train entry: freeze AR/FSQ, train point-cloud condition encoder."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AutoBrep point-cloud condition training")
    p.add_argument("--exp-dir", required=True)
    p.add_argument("--output-dir", default="")
    p.add_argument("--data-dir", default="/data/hdd/datasets/ABC-1M")
    p.add_argument("--datasplit", default="")
    p.add_argument("--dataset", default="abc-1m")
    p.add_argument("--task", default="gen")
    p.add_argument("--weight-folder", default="/data/hdd/outputs/AutoBrep")
    p.add_argument("--gpu", default="0")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--max-epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--limit-train", type=int, default=50000)
    p.add_argument("--limit-val", type=int, default=500)
    p.add_argument("--pc-num-points", type=int, default=2048)
    p.add_argument("--pc-num-latents", type=int, default=64)
    p.add_argument("--accumulate-grad-batches", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--resume-from", default="")
    p.add_argument("--config", default="")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo / "core" / "src"))

    exp_dir = Path(args.exp_dir)
    ckpt_dir = exp_dir / "checkpoints"
    tb_dir = exp_dir / "tensorboard"
    metrics_dir = exp_dir / "metrics"
    for d in (ckpt_dir, tb_dir, metrics_dir):
        d.mkdir(parents=True, exist_ok=True)

    weight = Path(args.weight_folder)
    config = Path(args.config) if args.config else repo / "configs" / "autobrep_pc.yaml"

    meta = {
        "mode": "train",
        "dataset": args.dataset,
        "task": args.task,
        "data_dir": args.data_dir,
        "weight_folder": str(weight),
        "batch_size": args.batch_size,
        "max_epochs": args.max_epochs,
        "lr": args.lr,
        "limit_train": args.limit_train,
        "pc_num_points": args.pc_num_points,
        "pc_num_latents": args.pc_num_latents,
        "freeze_backbone": True,
    }
    (metrics_dir / "train_config.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
    from pytorch_lightning.loggers import TensorBoardLogger
    import yaml

    with open(config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Apply CLI overrides onto yaml dict
    data_args = cfg["data"]["init_args"]
    data_args["data_root"] = args.data_dir
    data_args["batch_size"] = args.batch_size
    data_args["limit_train"] = args.limit_train
    data_args["limit_val"] = args.limit_val
    data_args["pc_num_points"] = args.pc_num_points
    data_args["num_workers"] = args.num_workers
    data_args["load_point_cloud"] = True

    model_args = cfg["model"]["init_args"]
    model_args["lr"] = args.lr
    model_args["pc_num_latents"] = args.pc_num_latents
    model_args["surf_fsq_ckpt"] = str(weight / "surf-fsq.ckpt")
    model_args["edge_fsq_ckpt"] = str(weight / "edge-fsq.ckpt")
    model_args["ar_ckpt"] = str(weight / "ar.ckpt")
    model_args["freeze_backbone"] = True

    from autobrep.data.abc_data import ARDataModule
    from autobrep.models.autoregressive import AutoBrepPCModel

    datamodule = ARDataModule(**data_args)
    model = AutoBrepPCModel(**model_args)

    ckpt_cb = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename="pc-cond-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )
    logger = TensorBoardLogger(save_dir=str(tb_dir), name="pc_cond")

    trainer_cfg = dict(cfg.get("trainer") or {})
    trainer_cfg.update(
        {
            "max_epochs": args.max_epochs,
            "accumulate_grad_batches": args.accumulate_grad_batches,
            "default_root_dir": str(exp_dir),
            "logger": logger,
            "callbacks": [ckpt_cb, LearningRateMonitor(logging_interval="step")],
        }
    )
    # Remove keys Lightning Trainer may not accept from yaml leftovers
    for k in ("profiler",):
        trainer_cfg.pop(k, None)

    trainer = Trainer(**trainer_cfg)
    ckpt_path = args.resume_from or None
    print(
        f"[train_pc] fit data={args.data_dir} batch={args.batch_size} "
        f"epochs={args.max_epochs} → {ckpt_dir}",
        file=sys.stderr,
    )
    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)

    summary = {
        "best_model_path": getattr(ckpt_cb, "best_model_path", ""),
        "best_model_score": float(ckpt_cb.best_model_score)
        if ckpt_cb.best_model_score is not None
        else None,
        "last_model_path": str(ckpt_dir / "last.ckpt"),
    }
    (metrics_dir / "train_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
