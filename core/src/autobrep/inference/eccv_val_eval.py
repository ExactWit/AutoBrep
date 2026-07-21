"""Official ECCV val: generate STEP from views+DXF, score with challenge eval.py."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

import torch
from pytorch_lightning import Callback, LightningModule, Trainer

from autobrep.data.token_mapping import MMTokenIndex
from autobrep.inference.brepgen_brep_builder import AutoBrepBuilder
from autobrep.inference.inference_common import reconstruct_compound
from autobrep.inference.view_condition import load_condition_for_sample
from autobrep.models.autoregressive import ARGenCheckpointPaths, AutoRegressiveSampler
from autobrep.models.vaes import EdgeFSQVAE, SurfaceFSQVAE
from occwl.io import save_step as save_step_func

DEFAULT_EVAL_PY = Path(
    "/data/hdd/datasets/eccv2026ws-cad-data/examples/min_eval/eval.py"
)


def load_official_val_ids(
    dataset_root: Path | str,
    *,
    max_samples: int = -1,
    split: str = "val",
    datasplit: str | Path | None = None,
    parquet_root: str | Path | None = None,
    require_gt: bool = True,
) -> list[str]:
    """
    Official datasplit ids for conditioned STEP eval.

    Prefer intersection with processed parquet split (has geometry that passed
    preprocess). ``max_samples <= 0`` → use all remaining ids.
    """
    root = Path(dataset_root)
    if datasplit:
        split_path = Path(datasplit)
    else:
        split_path = root / "processed" / "datasplit.json"
        if not split_path.is_file():
            split_path = root / "processed" / "autobrep" / "datasplit.json"
    data = json.loads(split_path.read_text(encoding="utf-8"))
    ids = [str(x) for x in (data.get("splits") or {}).get(split) or []]

    if parquet_root is not None:
        pq = Path(parquet_root) / split
        if pq.is_dir():
            import pyarrow.dataset as ds

            try:
                table = ds.dataset(str(pq), format="parquet").to_table(
                    columns=["sample_id"]
                )
                proc = {str(x) for x in table.column("sample_id").to_pylist()}
                ids = [i for i in ids if i in proc]
            except Exception as exc:  # noqa: BLE001
                print(f"[eccv_val] parquet id filter skipped: {exc}", flush=True)

    if require_gt:
        ids = [i for i in ids if gt_step_path(root, i).is_file()]

    if max_samples > 0:
        ids = ids[: int(max_samples)]
    return ids


def gt_step_path(dataset_root: Path, sample_id: str) -> Path:
    return Path(dataset_root) / "train" / "target_step" / f"{sample_id}.step"


def resolve_complexity_id(
    complexity: str,
    *,
    transformer: LightningModule | None = None,
    images: torch.Tensor | None = None,
    prim_types: torch.Tensor | None = None,
    prim_linetypes: torch.Tensor | None = None,
    prim_geom: torch.Tensor | None = None,
    prim_mask: torch.Tensor | None = None,
) -> tuple[int, str]:
    """
    Map complexity mode → token id.

    Modes:
      easy/medium/hard/random — fixed tokens (legacy infer)
      from_condition|auto|cond — AR next-token given view+DXF prepend
    """
    mode = str(complexity or "medium").lower().strip()
    fixed = {
        "easy": MMTokenIndex.GEN_EASY.value,
        "medium": MMTokenIndex.GEN_MID.value,
        "hard": MMTokenIndex.GEN_HARD.value,
        "random": MMTokenIndex.GEN_UNCOND.value,
        "uncond": MMTokenIndex.GEN_UNCOND.value,
    }
    if mode in ("from_condition", "auto", "cond"):
        if transformer is None or images is None or prim_types is None:
            raise ValueError("from_condition requires transformer + view/DXF tensors")
        predict = getattr(transformer, "predict_complexity_from_condition", None)
        if predict is None:
            raise TypeError(
                "transformer lacks predict_complexity_from_condition "
                "(need AutoBrepViewModel)"
            )
        cid = int(
            predict(
                images,
                prim_types,
                prim_linetypes,
                prim_geom,
                prim_mask,
            )
        )
        name = {
            MMTokenIndex.GEN_EASY.value: "easy",
            MMTokenIndex.GEN_MID.value: "medium",
            MMTokenIndex.GEN_HARD.value: "hard",
            MMTokenIndex.GEN_UNCOND.value: "random",
        }.get(cid, str(cid))
        return cid, f"from_condition:{name}"
    if mode in fixed:
        return fixed[mode], mode
    return MMTokenIndex.GEN_MID.value, f"fallback_medium({mode})"


def generate_pred_step(
    *,
    transformer: LightningModule,
    surface_fsq: SurfaceFSQVAE,
    edge_fsq: EdgeFSQVAE,
    device: torch.device,
    dataset_root: Path,
    sample_id: str,
    out_step: Path,
    complexity: str = "from_condition",
    temperature: float = 1.0,
    top_p: float = 0.9,
    vertex_threshold: float = 0.002,
    sewing_tolerance: float = 0.002,
    z_threshold: float = 0.0,
) -> dict[str, Any]:
    """Generate one STEP under view+DXF condition using the live train module."""
    images, dxf, meta = load_condition_for_sample(
        dataset_root, sample_id, batch_size=1
    )
    # Match vanilla AutoBrep sampler dtypes; disable outer AMP (PL bf16) which
    # interacts badly with AR generate under inference_mode.
    images = images.to(device=device, dtype=torch.float16)
    prim_types = dxf["prim_types"].to(device)
    prim_linetypes = dxf["prim_linetypes"].to(device)
    prim_geom = dxf["prim_geom"].to(device=device, dtype=torch.float16)
    prim_mask = dxf["prim_mask"].to(device)

    was_training = transformer.training
    transformer.eval()
    try:
        with torch.no_grad():
            autocast_ctx = (
                torch.autocast(device_type="cuda", enabled=False)
                if device.type == "cuda"
                else torch.autocast(device_type="cpu", enabled=False)
            )
            with autocast_ctx:
                complexity_id, complexity_resolved = resolve_complexity_id(
                    complexity,
                    transformer=transformer,
                    images=images,
                    prim_types=prim_types,
                    prim_linetypes=prim_linetypes,
                    prim_geom=prim_geom,
                    prim_mask=prim_mask,
                )
                prompt = (
                    torch.LongTensor(
                        [
                            MMTokenIndex.BOS.value,
                            MMTokenIndex.BOM.value,
                            complexity_id,
                            MMTokenIndex.EOM.value,
                            MMTokenIndex.BOC.value,
                        ]
                    )
                    .reshape(1, 5)
                    .to(device)
                )
                samples = transformer.generate(
                    prompt,
                    temperature,
                    top_p,
                    images=images,
                    prim_types=prim_types,
                    prim_linetypes=prim_linetypes,
                    prim_geom=prim_geom,
                    prim_mask=prim_mask,
                )
        tokens = torch.concat([prompt, samples], -1).detach().cpu().numpy()

        class _DecodeShim:
            pass

        shim = _DecodeShim()
        shim.transformer = transformer
        shim.surface_fsq = surface_fsq
        shim.edge_fsq = edge_fsq
        with torch.no_grad():
            decoded = AutoRegressiveSampler.decode_tokens(shim, tokens)
        cad_list = AutoRegressiveSampler.convert_to_cad_data(decoded) if decoded else []
        if not cad_list:
            return {
                "sample_id": sample_id,
                "ok": False,
                "error": "decode_failed",
                "complexity": complexity_resolved,
                "complexity_id": complexity_id,
                **meta,
            }

        builders = [
            AutoBrepBuilder(
                device=device,
                z_threshold=z_threshold,
                vertex_threshold=vertex_threshold,
                sewing_tolerance=sewing_tolerance,
            )
        ]
        # Lightning disables grads for the whole validation epoch; AutoBrepBuilder's
        # joint_optimize needs .backward() (same as scripts/infer_pipeline.py).
        with torch.enable_grad():
            compound = reconstruct_compound(cad_list[0], builders)
        if compound is None:
            return {
                "sample_id": sample_id,
                "ok": False,
                "error": "rebuild_failed",
                "complexity": complexity_resolved,
                "complexity_id": complexity_id,
                **meta,
            }

        out_step.parent.mkdir(parents=True, exist_ok=True)
        save_step_func([compound], out_step)
        return {
            "sample_id": sample_id,
            "ok": True,
            "step": str(out_step),
            "num_faces": int(cad_list[0].face_pos_cad.shape[0]),
            "complexity": complexity_resolved,
            "complexity_id": complexity_id,
            **meta,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "sample_id": sample_id,
            "ok": False,
            "error": f"exception:{type(exc).__name__}:{exc}",
            **(meta if "meta" in locals() else {}),
        }
    finally:
        transformer.train(was_training)


def run_official_eval(
    work_dir: Path,
    *,
    eval_py: Path = DEFAULT_EVAL_PY,
    timeout_sec: int = 3600,
) -> dict[str, Any]:
    """
    Run challenge ``eval.py`` with cwd=work_dir (expects gt/ and pred/).

    Parses the printed Summary block into a metric dict.
    """
    work_dir = Path(work_dir)
    eval_py = Path(eval_py)
    if not eval_py.is_file():
        raise FileNotFoundError(f"eval.py not found: {eval_py}")
    env = {
        **dict(os.environ),
        "EVAL_WORKERS": "1",
        "EVAL_MAX_WORKERS": "1",
        "EVAL_SAMPLE_TIMEOUT_SEC": "300",
    }
    proc = subprocess.run(
        [sys.executable, str(eval_py.resolve()), "pred"],
        cwd=str(work_dir),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout_sec,
        check=False,
    )
    out = (proc.stdout or "") + "\n" + (proc.stderr or "")
    metrics: dict[str, Any] = {"eval_returncode": float(proc.returncode)}
    patterns = {
        "valid_ratio": r"Valid Ratio:\s*([0-9.]+)",
        "surface_f1": r"Surface F1:\s*([0-9.]+)",
        "edge_f1": r"Edge F1:\s*([0-9.]+)",
        "vertex_f1": r"Vertex F1:\s*([0-9.]+)",
        "topo_f1": r"Topo F1:\s*([0-9.]+)",
        "summary": r"Summary:\s*([0-9.]+)",
        "valid_count": r"Valid Count:\s*([0-9]+)",
        "invalid_count": r"Invalid Count:\s*([0-9]+)",
    }
    for key, pat in patterns.items():
        m = re.search(pat, out)
        if m:
            metrics[key] = float(m.group(1))
    metrics["_raw_log_tail"] = out[-4000:]
    if "summary" not in metrics and proc.returncode != 0:
        metrics["summary"] = 0.0
    return metrics


def prepare_gt_pred_dirs(
    dataset_root: Path,
    sample_ids: Sequence[str],
    work_dir: Path,
) -> tuple[Path, Path, list[str]]:
    """Create work_dir/gt and work_dir/pred; copy GT STEPs. Returns (gt, pred, ok_ids)."""
    gt_dir = work_dir / "gt"
    pred_dir = work_dir / "pred"
    if gt_dir.exists():
        shutil.rmtree(gt_dir)
    if pred_dir.exists():
        shutil.rmtree(pred_dir)
    gt_dir.mkdir(parents=True)
    pred_dir.mkdir(parents=True)
    ok_ids: list[str] = []
    for sid in sample_ids:
        src = gt_step_path(dataset_root, sid)
        if not src.is_file():
            print(f"[eccv_val] skip missing GT STEP: {src}", flush=True)
            continue
        shutil.copy2(src, gt_dir / f"{sid}.step")
        ok_ids.append(sid)
    return gt_dir, pred_dir, ok_ids


class EccvOfficialValCallback(Callback):
    """
    After selected validation checks: generate STEP on official val ids,
    run challenge eval.py, log Surface/Edge/Vertex/Topo F1 + Summary to TB.
    """

    def __init__(
        self,
        *,
        dataset_root: str | Path,
        weight_folder: str | Path,
        sample_ids: Optional[Sequence[str]] = None,
        max_samples: int = -1,
        every_n_val_checks: int = 1,
        eval_py: str | Path = DEFAULT_EVAL_PY,
        work_dir: str | Path = "",
        datasplit: str | Path = "",
        parquet_root: str | Path = "",
        complexity: str = "from_condition",
        temperature: float = 1.0,
        top_p: float = 0.9,
        enabled: bool = True,
    ):
        super().__init__()
        self.dataset_root = Path(dataset_root)
        self.weight_folder = Path(weight_folder)
        self.max_samples = int(max_samples)
        self.every_n_val_checks = max(1, int(every_n_val_checks))
        self.eval_py = Path(eval_py)
        self.work_dir = Path(work_dir) if work_dir else None
        self.datasplit = Path(datasplit) if datasplit else None
        self.parquet_root = Path(parquet_root) if parquet_root else None
        self.complexity = complexity
        self.temperature = temperature
        self.top_p = top_p
        self.enabled = enabled
        self._sample_ids = list(sample_ids) if sample_ids else None
        self._val_check_count = 0
        self._surface_fsq: Optional[SurfaceFSQVAE] = None
        self._edge_fsq: Optional[EdgeFSQVAE] = None

    def _ensure_ids(self) -> list[str]:
        if self._sample_ids is None:
            self._sample_ids = load_official_val_ids(
                self.dataset_root,
                max_samples=self.max_samples,
                split="val",
                datasplit=self.datasplit,
                parquet_root=self.parquet_root,
                require_gt=True,
            )
        return list(self._sample_ids)

    def _ensure_fsq(self, device: torch.device) -> tuple[SurfaceFSQVAE, EdgeFSQVAE]:
        if self._surface_fsq is not None and self._edge_fsq is not None:
            return self._surface_fsq, self._edge_fsq
        paths = ARGenCheckpointPaths.from_folder(folder=str(self.weight_folder))
        self._surface_fsq = (
            SurfaceFSQVAE.load_from_checkpoint(paths.surface_fsq)
            .drop_encoder()
            .to(device)
            .eval()
        )
        self._edge_fsq = (
            EdgeFSQVAE.load_from_checkpoint(paths.edge_fsq)
            .drop_encoder()
            .to(device)
            .eval()
        )
        return self._surface_fsq, self._edge_fsq

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if not self.enabled:
            print("[eccv_val] disabled (--no-official-val)", flush=True)
            return
        sample_ids = self._ensure_ids()
        print(
            f"[eccv_val] callback ready: conditioned STEP gen on "
            f"official∩processed val (n={len(sample_ids)}, "
            f"max_samples={self.max_samples}, every_n={self.every_n_val_checks})",
            flush=True,
        )
        print(
            f"[eccv_val] first ids: {sample_ids[: min(8, len(sample_ids))]}",
            flush=True,
        )

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        if not self.enabled:
            return
        if trainer.sanity_checking:
            return
        self._val_check_count += 1
        if self._val_check_count % self.every_n_val_checks != 0:
            return

        sample_ids = self._ensure_ids()
        if not sample_ids:
            print("[eccv_val] no official val sample ids; skip", flush=True)
            return

        work = self.work_dir or (
            Path(trainer.default_root_dir)
            / "metrics"
            / f"official_val_step{int(trainer.global_step):06d}"
        )
        work.mkdir(parents=True, exist_ok=True)
        gt_dir, pred_dir, ok_ids = prepare_gt_pred_dirs(
            self.dataset_root, sample_ids, work
        )
        if not ok_ids:
            print("[eccv_val] no GT STEPs found; skip", flush=True)
            return

        print(
            f"[eccv_val] generating {len(ok_ids)} STEPs on official val "
            f"(step={trainer.global_step}) → {work}",
            flush=True,
        )
        was_training = pl_module.training
        device = next(pl_module.parameters()).device
        pl_module.eval()
        gen_log: list[dict[str, Any]] = []
        try:
            surface_fsq, edge_fsq = self._ensure_fsq(device)
            for i, sid in enumerate(ok_ids):
                try:
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    status = generate_pred_step(
                        transformer=pl_module,
                        surface_fsq=surface_fsq,
                        edge_fsq=edge_fsq,
                        device=device,
                        dataset_root=self.dataset_root,
                        sample_id=sid,
                        out_step=pred_dir / f"{sid}.step",
                        complexity=self.complexity,
                        temperature=self.temperature,
                        top_p=self.top_p,
                    )
                except Exception as exc:  # noqa: BLE001
                    status = {
                        "sample_id": sid,
                        "ok": False,
                        "error": f"exception:{exc}",
                    }
                gen_log.append(status)
                if (i + 1) % 10 == 0 or not status.get("ok") or i == 0:
                    print(
                        f"[eccv_val] [{i+1}/{len(ok_ids)}] {sid}: "
                        f"ok={status.get('ok')} err={status.get('error', '')}",
                        flush=True,
                    )
        finally:
            pl_module.train(was_training)

        n_ok = sum(1 for g in gen_log if g.get("ok"))
        for sid in list(ok_ids):
            if not (pred_dir / f"{sid}.step").is_file():
                (gt_dir / f"{sid}.step").unlink(missing_ok=True)

        metrics: dict[str, Any] = {
            "n_requested": len(ok_ids),
            "n_generated": n_ok,
            "gen_success_rate": n_ok / max(len(ok_ids), 1),
            "sample_ids": ok_ids,
            "gen_log": gen_log,
        }
        pred_steps = list(pred_dir.glob("*.step"))
        if pred_steps:
            try:
                official = run_official_eval(work, eval_py=self.eval_py)
                metrics["official"] = {
                    k: v for k, v in official.items() if not k.startswith("_")
                }
                metrics["official_log_tail"] = official.get("_raw_log_tail", "")[-2000:]
            except Exception as exc:  # noqa: BLE001
                print(f"[eccv_val] official eval failed: {exc}", flush=True)
                metrics["official_error"] = str(exc)
        else:
            metrics["official"] = {
                "summary": 0.0,
                "valid_ratio": 0.0,
                "surface_f1": 0.0,
                "edge_f1": 0.0,
                "vertex_f1": 0.0,
                "topo_f1": 0.0,
            }

        out_json = work / "metrics.json"
        out_json.write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        off = metrics.get("official") or {}
        log_map = {
            "val/official_gen_success": float(metrics["gen_success_rate"]),
            "val/official_summary": float(off.get("summary", 0.0)),
            "val/official_valid_ratio": float(off.get("valid_ratio", 0.0)),
            "val/official_surface_f1": float(off.get("surface_f1", 0.0)),
            "val/official_edge_f1": float(off.get("edge_f1", 0.0)),
            "val/official_vertex_f1": float(off.get("vertex_f1", 0.0)),
            "val/official_topo_f1": float(off.get("topo_f1", 0.0)),
        }
        for k, v in log_map.items():
            pl_module.log(k, v, prog_bar=(k.endswith("summary")), sync_dist=False)
            if trainer.logger is not None:
                trainer.logger.log_metrics({k: v}, step=trainer.global_step)

        print(
            f"[eccv_val] summary={log_map['val/official_summary']:.4f} "
            f"surf={log_map['val/official_surface_f1']:.4f} "
            f"edge={log_map['val/official_edge_f1']:.4f} "
            f"vert={log_map['val/official_vertex_f1']:.4f} "
            f"topo={log_map['val/official_topo_f1']:.4f} "
            f"gen_ok={n_ok}/{len(ok_ids)} → {out_json}",
            flush=True,
        )
