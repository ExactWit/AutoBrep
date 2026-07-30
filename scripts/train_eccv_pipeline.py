#!/usr/bin/env python3
"""exp_launcher train entry: freeze AR/FSQ, train multi-view condition encoder (ECCV)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AutoBrep ECCV view-conditioned SFT")
    p.add_argument("--exp-dir", required=True)
    p.add_argument("--output-dir", default="")
    p.add_argument("--data-dir", default="/data/hdd/datasets/eccv2026ws-cad-data")
    p.add_argument("--datasplit", default="")
    p.add_argument("--dataset", default="eccv2026ws-cad")
    p.add_argument("--task", default="gen")
    p.add_argument("--weight-folder", default="/data/hdd/outputs/AutoBrep")
    p.add_argument("--gpu", default="0")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--max-steps", type=int, default=10000)
    p.add_argument("--val-check-interval", type=int, default=500)
    p.add_argument(
        "--limit-val-batches",
        type=int,
        default=100,
        help="CE val batches per epoch (non-STEP); larger with bigger batch_size",
    )
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--view-num-latents", type=int, default=64)
    p.add_argument(
        "--use-prim-seq-encoder",
        type=int,
        default=1,
        help="P1-A: PrimTransformerEncoder + soft prefix compress (1=on)",
    )
    p.add_argument("--prim-d-model", type=int, default=512)
    p.add_argument("--prim-n-layers", type=int, default=4)
    p.add_argument("--prim-max-seq", type=int, default=384)
    p.add_argument(
        "--use-decoder-cross-attn",
        type=int,
        default=1,
        help="MM Stage A / P1-B: per-layer AR cross-attn (implies prim encoder)",
    )
    p.add_argument("--decoder-xattn-heads", type=int, default=8)
    p.add_argument(
        "--enable-aux-surf-type",
        type=int,
        default=1,
        help="Stage A: surface-type CE via SurfaceTypeHead (needs surf_type_ids in batch)",
    )
    p.add_argument("--aux-surf-type-weight", type=float, default=0.1)
    p.add_argument(
        "--enable-aux-view-bbox",
        type=int,
        default=0,
        help="Optional TechDraw AABB consistency aux loss",
    )
    p.add_argument("--aux-view-bbox-weight", type=float, default=0.1)
    p.add_argument(
        "--cond-cache-root",
        default="",
        help="If set: load images/techdraw/surf_type from processed/cond_cache_v2",
    )
    p.add_argument(
        "--ar-ckpt",
        default="",
        help="Parent AR Lightning ckpt to freeze (default: <weight-folder>/ar.ckpt)",
    )
    p.add_argument("--accumulate-grad-batches", type=int, default=2)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--resume-from", default="")
    p.add_argument("--config", default="")
    p.add_argument("--max-epochs", type=int, default=-1,
                   help=">0 时按 epoch 训练（会重复遍历小数据集）；忽略 --max-steps")
    p.add_argument("--limit-train", type=int, default=-1)
    p.add_argument("--limit-val", type=int, default=-1)
    # Official B-Rep quality eval (challenge min_eval/eval.py on datasplit val)
    p.add_argument(
        "--official-val-samples",
        type=int,
        default=-1,
        help="Full STEP eval sample count at final epoch; <=0 means ALL (~694)",
    )
    p.add_argument(
        "--official-val-samples-mid",
        type=int,
        default=24,
        help="Fixed STEP subset for mid milestones (25/50/75%%); keep small for speed",
    )
    p.add_argument(
        "--official-val-gen-batch",
        type=int,
        default=1,
        help="AR generate batch size during official STEP eval (use spare VRAM)",
    )
    p.add_argument(
        "--official-val-every",
        type=int,
        default=0,
        help="If >0: run STEP official eval every N epochs. "
        "If <=0: use --official-val-epoch-frac milestones (default).",
    )
    p.add_argument(
        "--official-val-epoch-frac",
        type=float,
        default=0.25,
        help="When official-val-every<=0: STEP eval at 25%%/50%%/75%%/100%% of max_epochs "
        "(CE val_loss still every epoch).",
    )
    p.add_argument(
        "--no-official-val",
        action="store_true",
        help="Disable STEP generation + challenge eval.py during training",
    )
    p.add_argument(
        "--eval-py",
        default="/data/hdd/datasets/eccv2026ws-cad-data/examples/min_eval/eval.py",
    )
    p.add_argument(
        "--complexity",
        default="from_condition",
        choices=[
            "from_condition",
            "auto",
            "cond",
            "easy",
            "medium",
            "hard",
            "random",
        ],
        help="Official-val / infer complexity token: from_condition=AR given view+DXF; "
        "easy/medium/hard/random=fixed (legacy). Train still uses GT face-count meta.",
    )
    return p.parse_args()


def resolve_eccv_paths(data_dir: str | Path) -> tuple[Path, Path]:
    """
    Returns (parquet_root, dataset_root).

    Accepts challenge root, ``.../processed``, or ``.../processed/autobrep``
    (exp_launcher typically passes the registry processed_path).
    """
    root = Path(data_dir).resolve()

    def _dataset_root_from(pq: Path) -> Path:
        # .../processed/autobrep → challenge root
        if pq.name == "autobrep" and pq.parent.name == "processed":
            return pq.parents[1]
        if pq.name == "processed":
            return pq.parent
        return pq

    # Already pointing at parquet splits
    if (root / "train").is_dir() and (root / "val").is_dir():
        # Distinguish challenge train/ (STEP) vs parquet train/
        if list((root / "train").glob("*.parquet")) or list(
            (root / "train").glob("**/*.parquet")
        ):
            return root, _dataset_root_from(root)

    # .../processed/autobrep
    if root.name == "autobrep" and (root / "train").is_dir():
        return root, _dataset_root_from(root)

    # .../processed → look for autobrep/
    if root.name == "processed":
        pq = root / "autobrep"
        if (pq / "train").is_dir() and (pq / "val").is_dir():
            return pq, root.parent
        # maybe parquet directly under processed (unlikely)
        if (root / "train").is_dir() and list((root / "train").glob("*.parquet")):
            return root, root.parent

    # Challenge root
    pq = root / "processed" / "autobrep"
    if (pq / "train").is_dir() and (pq / "val").is_dir():
        return pq, root

    raise FileNotFoundError(
        f"Cannot find ECCV AutoBrep parquet under {root}. "
        "Expected .../processed/autobrep/{train,val}. "
        "Run: ./run.sh preprocess --dataset eccv2026ws-cad-data --data-dir ..."
    )


def _require_cuda(gpu: str) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA unavailable after CUDA_VISIBLE_DEVICES={gpu!r}. "
            "Refusing to train on CPU."
        )
    torch.set_float32_matmul_precision("high")
    print(
        f"[train_eccv] CUDA ok: device={torch.cuda.get_device_name(0)} "
        f"visible={os.environ.get('CUDA_VISIBLE_DEVICES')}",
        file=sys.stderr,
    )


def main() -> int:
    args = parse_args()
    _require_cuda(args.gpu)

    # ECCV 数据量小：优先按 epoch 重复遍历；max_epochs>0 时关闭 max_steps。
    use_epochs = int(args.max_epochs) > 0
    if use_epochs:
        max_epochs = int(args.max_epochs)
        max_steps = -1
        schedule = "max_epochs"
    else:
        max_epochs = -1
        max_steps = int(args.max_steps)
        schedule = "max_steps"
        if args.limit_train > 0:
            print(
                "[train_eccv] NOTE: streaming IterableDataset; "
                f"--limit-train={args.limit_train} caps rows per epoch pass.",
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
    config = Path(args.config) if args.config else repo / "configs" / "autobrep_eccv.yaml"
    parquet_root, dataset_root = resolve_eccv_paths(args.data_dir)

    meta = {
        "mode": "train",
        "schedule": schedule,
        "conditioning": "view+dxf (no point cloud)",
        "dataset": args.dataset,
        "task": args.task,
        "data_dir": args.data_dir,
        "parquet_root": str(parquet_root),
        "dataset_root": str(dataset_root),
        "weight_folder": str(weight),
        "batch_size": args.batch_size,
        "max_steps": max_steps,
        "max_epochs": max_epochs,
        "accumulate_grad_batches": args.accumulate_grad_batches,
        "val_check_interval": args.val_check_interval,
        "limit_val_batches": args.limit_val_batches,
        "lr": args.lr,
        "view_num_latents": args.view_num_latents,
        "use_prim_seq_encoder": bool(args.use_prim_seq_encoder),
        "prim_d_model": args.prim_d_model,
        "prim_n_layers": args.prim_n_layers,
        "prim_max_seq": args.prim_max_seq,
        "use_decoder_cross_attn": bool(getattr(args, "use_decoder_cross_attn", 0)),
        "enable_aux_surf_type": bool(getattr(args, "enable_aux_surf_type", 1)),
        "aux_surf_type_weight": float(getattr(args, "aux_surf_type_weight", 0.1)),
        "cond_cache_root": str(getattr(args, "cond_cache_root", "") or ""),
        "freeze_backbone": True,
        "load_point_cloud": False,
        "loss": "AR token CE (prepend excluded)",
        "gpu": args.gpu,
        "cuda_device": torch.cuda.get_device_name(0),
        "official_val": not args.no_official_val,
        "official_val_samples": args.official_val_samples,
        "official_val_samples_mid": args.official_val_samples_mid,
        "official_val_gen_batch": args.official_val_gen_batch,
        "official_val_every": args.official_val_every,
        "official_val_epoch_frac": args.official_val_epoch_frac,
        "complexity": args.complexity,
        "eval_py": args.eval_py,
        "datasplit": args.datasplit
        or str(Path(dataset_root) / "processed" / "datasplit.json"),
        "tb_metrics": [
            "train_loss",
            "val_loss",
            "lr",
            "val/official_gen_success",
            "val/official_summary",
            "val/official_valid_ratio",
            "val/official_surface_f1",
            "val/official_edge_f1",
            "val/official_vertex_f1",
            "val/official_topo_f1",
        ],
    }
    (metrics_dir / "train_config.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    with open(config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_args = cfg["data"]["init_args"]
    data_args["data_root"] = str(parquet_root)
    data_args["dataset_root"] = str(dataset_root)
    data_args["batch_size"] = args.batch_size
    data_args["limit_train"] = None if args.limit_train <= 0 else args.limit_train
    data_args["limit_val"] = None if args.limit_val <= 0 else args.limit_val
    data_args["num_workers"] = args.num_workers
    data_args["load_point_cloud"] = False
    data_args["scaled_unique"] = False
    if getattr(args, "cond_cache_root", ""):
        data_args["cond_cache_root"] = str(args.cond_cache_root)

    model_args = cfg["model"]["init_args"]
    model_args["lr"] = args.lr
    model_args["view_num_latents"] = args.view_num_latents
    model_args["use_prim_seq_encoder"] = bool(args.use_prim_seq_encoder)
    model_args["prim_d_model"] = int(args.prim_d_model)
    model_args["prim_n_layers"] = int(args.prim_n_layers)
    model_args["prim_max_seq"] = int(args.prim_max_seq)
    model_args["use_decoder_cross_attn"] = bool(args.use_decoder_cross_attn)
    model_args["decoder_xattn_heads"] = int(args.decoder_xattn_heads)
    model_args["enable_aux_surf_type"] = bool(args.enable_aux_surf_type)
    model_args["aux_surf_type_weight"] = float(args.aux_surf_type_weight)
    model_args["enable_aux_view_bbox"] = bool(args.enable_aux_view_bbox)
    model_args["aux_view_bbox_weight"] = float(args.aux_view_bbox_weight)
    model_args["surf_fsq_ckpt"] = str(weight / "surf-fsq.ckpt")
    model_args["edge_fsq_ckpt"] = str(weight / "edge-fsq.ckpt")
    ar_ckpt = str(args.ar_ckpt).strip() if getattr(args, "ar_ckpt", "") else ""
    model_args["ar_ckpt"] = ar_ckpt or str(weight / "ar.ckpt")
    model_args["freeze_backbone"] = True

    from autobrep.data.eccv_data import ECCVViewDataModule
    from autobrep.inference.eccv_val_eval import EccvOfficialValCallback
    from autobrep.metrics.fast_val_callback import FastValMetricsCallback
    from autobrep.models.autoregressive import AutoBrepViewModel

    datamodule = ECCVViewDataModule(**data_args)
    datamodule.setup("fit")
    train_rows = len(datamodule._train_ds) if datamodule._train_ds is not None else -1
    val_rows = len(datamodule._val_ds) if datamodule._val_ds is not None else -1
    steps_per_epoch = max(
        1,
        (train_rows + args.batch_size - 1) // max(1, args.batch_size),
    )
    datasplit_path = (
        Path(args.datasplit)
        if args.datasplit
        else Path(dataset_root) / "processed" / "datasplit.json"
    )
    print(
        f"[train_eccv] parquet val = official datasplit val "
        f"(datasplit={datasplit_path}, exists={datasplit_path.is_file()})",
        file=sys.stderr,
    )
    print(
        f"[train_eccv] epoch size: train_rows={train_rows} val_rows={val_rows} "
        f"batch={args.batch_size} → ~{steps_per_epoch} train batches/epoch "
        f"(accum={args.accumulate_grad_batches} → "
        f"~{steps_per_epoch // max(1, args.accumulate_grad_batches)} opt steps/epoch)",
        file=sys.stderr,
    )
    meta["train_rows"] = train_rows
    meta["val_rows"] = val_rows
    meta["steps_per_epoch"] = steps_per_epoch
    meta["opt_steps_per_epoch_est"] = steps_per_epoch // max(
        1, args.accumulate_grad_batches
    )
    (metrics_dir / "train_config.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        "[train_eccv] constructing model (load ckpt → then move to CUDA)...",
        file=sys.stderr,
    )
    model = AutoBrepViewModel(**model_args)
    model = model.cuda()
    print(
        f"[train_eccv] model on {next(model.parameters()).device}, "
        f"allocated={torch.cuda.memory_allocated()/1e9:.2f}GB",
        file=sys.stderr,
    )

    from autobrep.diagnostics.model_info import log_model_info

    log_model_info(
        model,
        out_path=str(metrics_dir / "model_info.json"),
        stream=sys.stderr,
        max_depth=2,
    )
    # thop may leave total_ops/total_params buffers that break resume load_state_dict
    for mod in model.modules():
        for name in ("total_ops", "total_params"):
            if name in getattr(mod, "_buffers", {}):
                del mod._buffers[name]

    class _CudaWatch(Callback):
        def on_train_start(self, trainer, pl_module):
            dev = next(pl_module.parameters()).device
            if dev.type != "cuda":
                raise RuntimeError(f"Training on {dev}, expected CUDA")
            print(
                f"[train_eccv] on_train_start device={dev} "
                f"max_epochs={trainer.max_epochs} max_steps={trainer.max_steps} "
                f"allocated={torch.cuda.memory_allocated()/1e9:.2f}GB",
                file=sys.stderr,
                flush=True,
            )

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            step = int(trainer.global_step)
            if step <= 1 or step % 50 == 0:
                print(
                    f"[train_eccv] global_step={step} "
                    f"alloc={torch.cuda.memory_allocated()/1e9:.2f}GB "
                    f"peak={torch.cuda.max_memory_allocated()/1e9:.2f}GB",
                    file=sys.stderr,
                    flush=True,
                )

    ckpt_cb = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename="eccv-view-{step:06d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        every_n_train_steps=None,
    )
    logger = TensorBoardLogger(save_dir=str(tb_dir), name="eccv_view")

    official_val_cb = EccvOfficialValCallback(
        dataset_root=dataset_root,
        weight_folder=weight,
        max_samples=args.official_val_samples,
        max_samples_mid=args.official_val_samples_mid,
        gen_batch_size=args.official_val_gen_batch,
        every_n_val_checks=args.official_val_every,
        epoch_frac=args.official_val_epoch_frac,
        max_epochs=max_epochs if max_epochs > 0 else 50,
        eval_py=args.eval_py,
        datasplit=datasplit_path if datasplit_path.is_file() else "",
        parquet_root=parquet_root,
        complexity=args.complexity,
        enabled=not args.no_official_val,
    )
    fast_val_cb = FastValMetricsCallback(metrics_dir=metrics_dir, enabled=True)

    trainer_cfg = dict(cfg.get("trainer") or {})
    trainer_cfg.update(
        {
            "accelerator": "gpu",
            "devices": [0],
            "default_root_dir": str(exp_dir),
            "logger": logger,
            "accumulate_grad_batches": args.accumulate_grad_batches,
            "limit_val_batches": int(args.limit_val_batches),
            "callbacks": [
                ckpt_cb,
                LearningRateMonitor(logging_interval="step"),
                _CudaWatch(),
                fast_val_cb,
                official_val_cb,
            ],
        }
    )
    if use_epochs:
        # IterableDataset: each epoch re-scans parquet (repeat passes on small ECCV set).
        trainer_cfg["max_epochs"] = max_epochs
        trainer_cfg["max_steps"] = -1
        trainer_cfg["check_val_every_n_epoch"] = 1
        trainer_cfg.pop("val_check_interval", None)
    else:
        trainer_cfg["max_epochs"] = -1
        trainer_cfg["max_steps"] = max_steps
        trainer_cfg["val_check_interval"] = int(args.val_check_interval)
        trainer_cfg.pop("check_val_every_n_epoch", None)
    for k in ("profiler",):
        trainer_cfg.pop(k, None)

    trainer = Trainer(**trainer_cfg)
    ckpt_path = args.resume_from or None
    if ckpt_path:
        # Sanitize legacy ckpts that contain thop profiling buffers.
        raw = Path(ckpt_path)
        if raw.is_file():
            blob = torch.load(raw, map_location="cpu", weights_only=False)
            sd = blob.get("state_dict")
            if isinstance(sd, dict):
                cleaned = {
                    k: v
                    for k, v in sd.items()
                    if not k.endswith("total_ops") and not k.endswith("total_params")
                }
                if len(cleaned) != len(sd):
                    blob["state_dict"] = cleaned
                    sanitized = exp_dir / "checkpoints" / f"{raw.stem}.sanitized.ckpt"
                    sanitized.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(blob, sanitized)
                    ckpt_path = str(sanitized)
                    print(
                        f"[train_eccv] stripped thop buffers from resume ckpt → {sanitized}",
                        file=sys.stderr,
                        flush=True,
                    )
    print(
        f"[train_eccv] fit parquet={parquet_root} dataset_root={dataset_root} "
        f"cond=view+dxf (no PC) batch={args.batch_size} "
        f"schedule={schedule} max_epochs={max_epochs} max_steps={max_steps} "
        f"accum={args.accumulate_grad_batches} official_val={not args.no_official_val} "
        f"resume={ckpt_path or ''} → {ckpt_dir}",
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
        "max_steps": max_steps,
        "max_epochs": max_epochs,
        "schedule": schedule,
        "global_step": int(trainer.global_step),
        "current_epoch": int(trainer.current_epoch),
    }
    (metrics_dir / "train_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
