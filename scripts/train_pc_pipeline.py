#!/usr/bin/env python3
"""exp_launcher train entry: freeze AR/FSQ, train point-cloud condition encoder.

Streaming parquet (IterableDataset) → stop by optimizer ``max_steps``, not epochs.
"""

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
    # Step-based controls (preferred for streaming)
    p.add_argument(
        "--max-steps",
        type=int,
        default=10000,
        help="Optimizer steps (gradient updates). Primary stop condition.",
    )
    p.add_argument(
        "--val-check-interval",
        type=int,
        default=500,
        help="Run validation every N training batches.",
    )
    p.add_argument(
        "--limit-val-batches",
        type=int,
        default=50,
        help="Cap val microbatches per check (streaming val has no epoch end).",
    )
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--pc-num-points", type=int, default=2048)
    p.add_argument("--pc-num-latents", type=int, default=64)
    p.add_argument("--accumulate-grad-batches", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--resume-from", default="")
    p.add_argument("--config", default="")
    # Legacy aliases (ignored with a warning) so old launcher payloads don't crash
    p.add_argument("--max-epochs", type=int, default=-1)
    p.add_argument("--limit-train", type=int, default=-1)
    p.add_argument("--limit-val", type=int, default=-1)
    return p.parse_args()


def resolve_parquet_data_root(data_dir: str | Path) -> Path:
    """
    AutoBrep expects ``{root}/train|val|test/*.parquet``.

    exp_launcher often passes the datasplit parent
    ``.../ABC-1M/processed``; parquet splits live one level up.
    """
    root = Path(data_dir).resolve()
    if (root / "train").is_dir() and (root / "val").is_dir():
        return root
    parent = root.parent
    if (parent / "train").is_dir() and (parent / "val").is_dir():
        print(
            f"[train_pc] data-dir={root} has no train/val; using parent {parent}",
            file=sys.stderr,
        )
        return parent
    raise FileNotFoundError(
        f"Cannot find train/val parquet under {root} or {parent}. "
        "Expected ABC-1M layout with train/ val/ test/."
    )


def _require_cuda(gpu: str) -> None:
    # Must set before importing torch.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA unavailable after CUDA_VISIBLE_DEVICES={gpu!r}. "
            "Refusing to train on CPU."
        )
    torch.set_float32_matmul_precision("high")
    print(
        f"[train_pc] CUDA ok: device={torch.cuda.get_device_name(0)} "
        f"visible={os.environ.get('CUDA_VISIBLE_DEVICES')}",
        file=sys.stderr,
    )


def main() -> int:
    args = parse_args()
    _require_cuda(args.gpu)

    if args.max_epochs > 0 or args.limit_train > 0:
        print(
            "[train_pc] NOTE: streaming run ignores --max-epochs/--limit-train; "
            f"using --max-steps={args.max_steps} (optimizer updates).",
            file=sys.stderr,
        )

    import torch
    from pytorch_lightning import Trainer, Callback
    from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
    from pytorch_lightning.loggers import TensorBoardLogger
    import yaml

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
    data_root = resolve_parquet_data_root(args.data_dir)

    micro_batches = args.max_steps * max(1, args.accumulate_grad_batches)
    meta = {
        "mode": "train",
        "schedule": "max_steps",
        "dataset": args.dataset,
        "task": args.task,
        "data_dir": args.data_dir,
        "data_root": str(data_root),
        "weight_folder": str(weight),
        "batch_size": args.batch_size,
        "max_steps": args.max_steps,
        "accumulate_grad_batches": args.accumulate_grad_batches,
        "approx_train_microbatches": micro_batches,
        "val_check_interval": args.val_check_interval,
        "limit_val_batches": args.limit_val_batches,
        "lr": args.lr,
        "pc_num_points": args.pc_num_points,
        "pc_num_latents": args.pc_num_latents,
        "freeze_backbone": True,
        "gpu": args.gpu,
        "cuda_device": torch.cuda.get_device_name(0),
    }
    (metrics_dir / "train_config.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    with open(config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_args = cfg["data"]["init_args"]
    data_args["data_root"] = str(data_root)
    data_args["batch_size"] = args.batch_size
    # Unbounded stream; Trainer.max_steps stops training.
    data_args["limit_train"] = None
    data_args["limit_val"] = None
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
    print("[train_pc] constructing model (load ckpt → then move to CUDA)...", file=sys.stderr)
    model = AutoBrepPCModel(**model_args)
    model = model.cuda()
    print(
        f"[train_pc] model on {next(model.parameters()).device}, "
        f"allocated={torch.cuda.memory_allocated()/1e9:.2f}GB",
        file=sys.stderr,
    )

    class _CudaWatch(Callback):
        def on_train_start(self, trainer, pl_module):
            dev = next(pl_module.parameters()).device
            if dev.type != "cuda":
                raise RuntimeError(f"Training on {dev}, expected CUDA")
            print(
                f"[train_pc] on_train_start device={dev} "
                f"max_steps={trainer.max_steps} "
                f"allocated={torch.cuda.memory_allocated()/1e9:.2f}GB "
                f"reserved={torch.cuda.memory_reserved()/1e9:.2f}GB",
                file=sys.stderr,
                flush=True,
            )

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            step = int(trainer.global_step)
            if step <= 1 or step % 50 == 0:
                print(
                    f"[train_pc] global_step={step} "
                    f"alloc={torch.cuda.memory_allocated()/1e9:.2f}GB "
                    f"peak={torch.cuda.max_memory_allocated()/1e9:.2f}GB",
                    file=sys.stderr,
                    flush=True,
                )

    ckpt_cb = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename="pc-cond-step{step:06d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        every_n_train_steps=None,  # save when val runs
    )
    logger = TensorBoardLogger(save_dir=str(tb_dir), name="pc_cond")

    trainer_cfg = dict(cfg.get("trainer") or {})
    trainer_cfg.update(
        {
            "accelerator": "gpu",
            "devices": [0],
            "max_epochs": -1,  # unused; streaming is step-limited
            "max_steps": int(args.max_steps),
            "val_check_interval": int(args.val_check_interval),
            "limit_val_batches": int(args.limit_val_batches),
            "accumulate_grad_batches": args.accumulate_grad_batches,
            "default_root_dir": str(exp_dir),
            "logger": logger,
            "callbacks": [
                ckpt_cb,
                LearningRateMonitor(logging_interval="step"),
                _CudaWatch(),
            ],
        }
    )
    for k in ("profiler", "check_val_every_n_epoch"):
        trainer_cfg.pop(k, None)

    trainer = Trainer(**trainer_cfg)
    ckpt_path = args.resume_from or None
    print(
        f"[train_pc] fit data_root={data_root} batch={args.batch_size} "
        f"max_steps={args.max_steps} (opt updates) "
        f"accum={args.accumulate_grad_batches} "
        f"≈{micro_batches} microbatches → {ckpt_dir}",
        file=sys.stderr,
    )
    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)

    summary = {
        "best_model_path": getattr(ckpt_cb, "best_model_path", ""),
        "best_model_score": float(ckpt_cb.best_model_score)
        if ckpt_cb.best_model_score is not None
        else None,
        "last_model_path": str(ckpt_dir / "last.ckpt"),
        "peak_cuda_GB": round(torch.cuda.max_memory_allocated() / 1e9, 3),
        "max_steps": args.max_steps,
        "global_step": int(trainer.global_step),
    }
    (metrics_dir / "train_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
