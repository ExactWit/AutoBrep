"""Official ECCV val: generate STEP from views+DXF, score with challenge eval.py."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
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
    prim_group_ids: torch.Tensor | None = None,
) -> tuple[list[int], list[str]]:
    """
    Map complexity mode → token id(s). Always returns per-batch lists.
    """
    mode = str(complexity or "medium").lower().strip()
    fixed = {
        "easy": MMTokenIndex.GEN_EASY.value,
        "medium": MMTokenIndex.GEN_MID.value,
        "hard": MMTokenIndex.GEN_HARD.value,
        "random": MMTokenIndex.GEN_UNCOND.value,
        "uncond": MMTokenIndex.GEN_UNCOND.value,
    }
    b = 1 if images is None else int(images.shape[0])
    name_of = {
        MMTokenIndex.GEN_EASY.value: "easy",
        MMTokenIndex.GEN_MID.value: "medium",
        MMTokenIndex.GEN_HARD.value: "hard",
        MMTokenIndex.GEN_UNCOND.value: "random",
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
        raw = predict(
            images,
            prim_types,
            prim_linetypes,
            prim_geom,
            prim_mask,
            prim_group_ids=prim_group_ids,
        )
        if isinstance(raw, list):
            cids = [int(x) for x in raw]
        else:
            cids = [int(raw)] * b if b > 1 else [int(raw)]
        if len(cids) == 1 and b > 1:
            cids = cids * b
        labels = [f"from_condition:{name_of.get(c, str(c))}" for c in cids]
        return cids, labels
    if mode in fixed:
        cid = fixed[mode]
        return [cid] * b, [mode] * b
    return [MMTokenIndex.GEN_MID.value] * b, [f"fallback_medium({mode})"] * b


def _stack_conditions(
    dataset_root: Path,
    sample_ids: Sequence[str],
    device: torch.device,
    *,
    split: str = "val",
) -> tuple[torch.Tensor, dict[str, torch.Tensor], list[dict[str, Any]]]:
    """Load and stack conditions for distinct sample ids → batch dim B."""
    images_list = []
    dxf_lists: dict[str, list[torch.Tensor]] = {}
    metas: list[dict[str, Any]] = []
    for sid in sample_ids:
        images, dxf, meta = load_condition_for_sample(
            dataset_root, sid, batch_size=1, split=split
        )
        images_list.append(images[0])
        metas.append(meta)
        for k, v in dxf.items():
            if not isinstance(v, torch.Tensor):
                continue
            # load_condition_for_sample returns batched (1, ...) tensors; strip
            # the leading batch dim before stacking, else we get (B, 1, ...).
            if v.ndim >= 1 and int(v.shape[0]) == 1:
                v = v[0]
            dxf_lists.setdefault(k, []).append(v)
    images_b = torch.stack(images_list, dim=0).to(device=device, dtype=torch.float16)
    dxf_b: dict[str, torch.Tensor] = {}
    for k, vs in dxf_lists.items():
        stacked = torch.stack(vs, dim=0)
        if k in {"prim_geom"}:
            stacked = stacked.to(device=device, dtype=torch.float16)
        else:
            stacked = stacked.to(device=device)
        dxf_b[k] = stacked
    return images_b, dxf_b, metas


def _tokens_to_step(
    *,
    tokens_1d,
    transformer: LightningModule,
    surface_fsq: SurfaceFSQVAE,
    edge_fsq: EdgeFSQVAE,
    device: torch.device,
    sample_id: str,
    out_step: Path,
    complexity_resolved: str,
    complexity_id: int,
    meta: dict[str, Any],
    vertex_threshold: float,
    sewing_tolerance: float,
    z_threshold: float,
) -> dict[str, Any]:
    class _DecodeShim:
        pass

    shim = _DecodeShim()
    shim.transformer = transformer
    shim.surface_fsq = surface_fsq
    shim.edge_fsq = edge_fsq
    with torch.no_grad():
        decoded = AutoRegressiveSampler.decode_tokens(
            shim, tokens_1d.reshape(1, -1)
        )
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


def generate_pred_steps_batched(
    *,
    transformer: LightningModule,
    surface_fsq: SurfaceFSQVAE,
    edge_fsq: EdgeFSQVAE,
    device: torch.device,
    dataset_root: Path,
    sample_ids: Sequence[str],
    pred_dir: Path,
    complexity: str = "from_condition",
    temperature: float = 1.0,
    top_p: float = 0.9,
    gen_batch_size: int = 1,
    vertex_threshold: float = 0.002,
    sewing_tolerance: float = 0.002,
    z_threshold: float = 0.0,
    split: str = "val",
    require_gt: bool = True,
    gen_retries: int = 1,
    gen_rerank: bool = False,
) -> list[dict[str, Any]]:
    """
    Batched AR generate (uses more VRAM), then serial decode/rebuild per sample.

    ``gen_batch_size`` controls how many conditioned prompts share one AR pass.
    ``split`` selects condition root (``public_test`` → ``test_public/``).
    ``gen_retries`` > 1 resamples failed samples (decode/rebuild failure) up to
    that many total attempts; temperature/top_p sampling gives fresh candidates.
    """
    was_training = transformer.training
    transformer.eval()
    ids = [str(s) for s in sample_ids]
    _ = require_gt  # reserved; callers filter ids before invoking
    n_total = len(ids)
    bs = max(1, int(gen_batch_size))
    max_attempts = max(1, int(gen_retries))
    do_rerank = bool(gen_rerank) and max_attempts > 1
    t_all = time.perf_counter()
    final: dict[str, dict[str, Any]] = {}
    pending = list(ids)
    n_ok_total = 0

    def _rerank_score(status: dict[str, Any]) -> float:
        if not status.get("ok"):
            return -1e9
        nf = int(status.get("num_faces") or 0)
        label = str(status.get("complexity") or "").replace("from_condition:", "")
        if label == "easy":
            in_bucket = 1.0 if nf < 25 else 0.0
        elif label == "hard":
            in_bucket = 1.0 if nf >= 50 else 0.0
        else:
            in_bucket = 1.0 if 25 <= nf < 50 else (0.5 if nf < 25 else 0.0)
        return in_bucket * 1000.0 + float(nf)

    print(
        f"[STEP gen] start n={n_total} gen_batch={bs} "
        f"split={split} complexity={complexity} retries={max_attempts} "
        f"rerank={int(do_rerank)}",
        flush=True,
    )
    try:
      for attempt in range(1, max_attempts + 1):
        if not pending and not do_rerank:
            break
        if do_rerank:
            ids_this = list(ids)
        else:
            ids_this = pending
        pending = []
        n_round = len(ids_this)
        n_chunks = (n_round + bs - 1) // bs if n_round else 0
        n_ok = 0
        n_fail = 0
        if max_attempts > 1:
            print(f"[STEP gen] attempt {attempt}/{max_attempts} n={n_round}", flush=True)
        for chunk_i, start in enumerate(range(0, n_round, bs)):
            chunk = ids_this[start : start + bs]
            t_chunk = time.perf_counter()
            t_ar = 0.0
            t_dec = 0.0
            try:
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                images, dxf, metas = _stack_conditions(
                    dataset_root, chunk, device, split=split
                )
                with torch.no_grad():
                    autocast_ctx = (
                        torch.autocast(device_type="cuda", enabled=False)
                        if device.type == "cuda"
                        else torch.autocast(device_type="cpu", enabled=False)
                    )
                    with autocast_ctx:
                        cids, clabels = resolve_complexity_id(
                            complexity,
                            transformer=transformer,
                            images=images,
                            prim_types=dxf["prim_types"],
                            prim_linetypes=dxf["prim_linetypes"],
                            prim_geom=dxf["prim_geom"],
                            prim_mask=dxf["prim_mask"],
                            prim_group_ids=dxf.get("prim_group_ids"),
                        )
                        prompt_rows = []
                        for c in cids:
                            prompt_rows.extend(
                                [
                                    MMTokenIndex.BOS.value,
                                    MMTokenIndex.BOM.value,
                                    int(c),
                                    MMTokenIndex.EOM.value,
                                    MMTokenIndex.BOC.value,
                                ]
                            )
                        prompt = (
                            torch.tensor(prompt_rows, dtype=torch.long, device=device)
                            .view(len(chunk), 5)
                        )
                        t_ar0 = time.perf_counter()
                        samples = transformer.generate(
                            prompt,
                            temperature,
                            top_p,
                            images=images,
                            prim_types=dxf["prim_types"],
                            prim_linetypes=dxf["prim_linetypes"],
                            prim_geom=dxf["prim_geom"],
                            prim_mask=dxf["prim_mask"],
                            prim_group_ids=dxf.get("prim_group_ids"),
                        )
                        t_ar = time.perf_counter() - t_ar0
                        tokens = torch.concat([prompt, samples], -1).detach().cpu()

                t_dec0 = time.perf_counter()
                for i, sid in enumerate(chunk):
                    try:
                        status = _tokens_to_step(
                            tokens_1d=tokens[i].numpy(),
                            transformer=transformer,
                            surface_fsq=surface_fsq,
                            edge_fsq=edge_fsq,
                            device=device,
                            sample_id=sid,
                            out_step=pred_dir / f"{sid}.step",
                            complexity_resolved=clabels[i],
                            complexity_id=int(cids[i]),
                            meta=metas[i],
                            vertex_threshold=vertex_threshold,
                            sewing_tolerance=sewing_tolerance,
                            z_threshold=z_threshold,
                        )
                    except Exception as exc:  # noqa: BLE001
                        status = {
                            "sample_id": sid,
                            "ok": False,
                            "error": f"exception:{type(exc).__name__}:{exc}",
                        }
                    status["attempts"] = attempt
                    if status.get("ok"):
                        n_ok += 1
                        prev = final.get(sid)
                        if do_rerank and prev is not None and prev.get("ok"):
                            if _rerank_score(status) > _rerank_score(prev):
                                # Overwrite STEP file already written by _tokens_to_step.
                                final[sid] = status
                        else:
                            final[sid] = status
                    else:
                        n_fail += 1
                        if sid not in final or not final[sid].get("ok"):
                            final[sid] = status
                        if (not do_rerank) and attempt < max_attempts:
                            pending.append(sid)
                t_dec = time.perf_counter() - t_dec0
            except Exception as exc:  # noqa: BLE001
                for sid in chunk:
                    status = {
                        "sample_id": sid,
                        "ok": False,
                        "error": f"exception:{type(exc).__name__}:{exc}",
                        "attempts": attempt,
                    }
                    if sid not in final or not final[sid].get("ok"):
                        final[sid] = status
                    n_fail += 1
                    if (not do_rerank) and attempt < max_attempts:
                        pending.append(sid)
            done = n_ok_total + n_ok + n_fail
            elapsed = max(time.perf_counter() - t_all, 1e-6)
            rate_h = done / elapsed * 3600.0
            eta_s = (n_total - done) / max(done / elapsed, 1e-9) if done else float("nan")
            eta_h = eta_s / 3600.0 if done else float("nan")
            chunk_s = time.perf_counter() - t_chunk
            print(
                f"[STEP gen] {done}/{n_total} ({100.0 * done / max(n_total, 1):.1f}%) "
                f"ok={n_ok_total + n_ok} fail={n_fail} | {rate_h:.1f}/h ETA {eta_h:.1f}h | "
                f"attempt {attempt}/{max_attempts} chunk {chunk_i + 1}/{n_chunks} {chunk_s:.1f}s "
                f"(AR {t_ar:.1f}s + decode {t_dec:.1f}s) ids={chunk}",
                flush=True,
            )
        n_ok_total += n_ok
    finally:
        transformer.train(was_training)
    results = [final[sid] for sid in ids]
    n_ok_final = sum(1 for r in results if r.get("ok"))
    elapsed = time.perf_counter() - t_all
    print(
        f"[STEP gen] done {n_total}/{n_total} ok={n_ok_final} fail={n_total - n_ok_final} "
        f"attempts_used<={max_attempts} "
        f"in {elapsed / 60.0:.1f} min ({n_total / max(elapsed, 1e-6) * 3600.0:.1f}/h)",
        flush=True,
    )
    return results


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
    """Generate one STEP (thin wrapper over batched path)."""
    logs = generate_pred_steps_batched(
        transformer=transformer,
        surface_fsq=surface_fsq,
        edge_fsq=edge_fsq,
        device=device,
        dataset_root=dataset_root,
        sample_ids=[sample_id],
        pred_dir=out_step.parent,
        complexity=complexity,
        temperature=temperature,
        top_p=top_p,
        gen_batch_size=1,
        vertex_threshold=vertex_threshold,
        sewing_tolerance=sewing_tolerance,
        z_threshold=z_threshold,
    )
    status = logs[0] if logs else {"sample_id": sample_id, "ok": False, "error": "empty"}
    # ensure out path name if ok
    if status.get("ok") and out_step.name != f"{sample_id}.step":
        src = Path(status["step"])
        if src.is_file() and src.resolve() != out_step.resolve():
            out_step.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, out_step)
            status["step"] = str(out_step)
    return status


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
    After selected validation epochs: generate STEP on official val ids,
    run challenge eval.py, log Surface/Edge/Vertex/Topo F1 + Summary to TB.

    CE ``val_loss`` still runs every Lightning val epoch; STEP gen is sparse
    (default: at 25%/50%/75%/100% of ``max_epochs``).
    """

    def __init__(
        self,
        *,
        dataset_root: str | Path,
        weight_folder: str | Path,
        sample_ids: Optional[Sequence[str]] = None,
        max_samples: int = -1,
        max_samples_mid: int = 24,
        gen_batch_size: int = 1,
        every_n_val_checks: int = 0,
        epoch_frac: float = 0.25,
        max_epochs: int = 50,
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
        self.max_samples_mid = int(max_samples_mid)
        self.gen_batch_size = max(1, int(gen_batch_size))
        # >0: every N completed epochs; <=0: use epoch_frac milestones
        self.every_n_val_checks = int(every_n_val_checks)
        self.epoch_frac = float(epoch_frac)
        self.max_epochs = int(max_epochs)
        self.eval_py = Path(eval_py)
        self.work_dir = Path(work_dir) if work_dir else None
        self.datasplit = Path(datasplit) if datasplit else None
        self.parquet_root = Path(parquet_root) if parquet_root else None
        self.complexity = complexity
        self.temperature = temperature
        self.top_p = top_p
        self.enabled = enabled
        self._sample_ids = list(sample_ids) if sample_ids else None
        self._mid_ids: Optional[list[str]] = None
        self._val_check_count = 0
        self._surface_fsq: Optional[SurfaceFSQVAE] = None
        self._edge_fsq: Optional[EdgeFSQVAE] = None
        self._milestones = self._compute_milestones()

    def _compute_milestones(self) -> set[int]:
        """1-based epoch indices that should run official STEP eval."""
        if self.every_n_val_checks > 0:
            return set()
        me = max(1, self.max_epochs)
        frac = self.epoch_frac if self.epoch_frac > 0 else 0.25
        marks: set[int] = set()
        k = 1
        while k * frac < 1.0 - 1e-9:
            marks.add(max(1, int(round(k * frac * me))))
            k += 1
        marks.add(me)
        return marks

    def _should_run_official(self, trainer: Trainer) -> bool:
        # Lightning current_epoch is 0-based for the epoch that just validated.
        epoch_1based = int(trainer.current_epoch) + 1
        if self.every_n_val_checks > 0:
            return epoch_1based % self.every_n_val_checks == 0
        me = int(trainer.max_epochs) if int(trainer.max_epochs or 0) > 0 else self.max_epochs
        if me != self.max_epochs:
            self.max_epochs = me
            self._milestones = self._compute_milestones()
        return epoch_1based in self._milestones

    def _ensure_ids(self) -> list[str]:
        if self._sample_ids is None:
            # Full pool (ignore max_samples here; slice later per mid/full).
            self._sample_ids = load_official_val_ids(
                self.dataset_root,
                max_samples=-1,
                split="val",
                datasplit=self.datasplit,
                parquet_root=self.parquet_root,
                require_gt=True,
            )
        if self._mid_ids is None:
            n_mid = self.max_samples_mid
            if n_mid <= 0:
                n_mid = 24
            self._mid_ids = list(self._sample_ids[:n_mid])
        return list(self._sample_ids)

    def _ids_for_eval(self, trainer: Trainer) -> tuple[list[str], str]:
        """Mid milestones → fixed small set; final epoch → full val."""
        all_ids = self._ensure_ids()
        epoch_1based = int(trainer.current_epoch) + 1
        is_final = epoch_1based >= max(1, self.max_epochs)
        if is_final:
            if self.max_samples > 0:
                return all_ids[: self.max_samples], "full"
            return all_ids, "full"
        return list(self._mid_ids or all_ids[:24]), "mid"

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
        me = int(trainer.max_epochs) if int(trainer.max_epochs or 0) > 0 else self.max_epochs
        self.max_epochs = me
        self._milestones = self._compute_milestones()
        self._ensure_ids()
        if self.every_n_val_checks > 0:
            sched = f"every {self.every_n_val_checks} epoch(s)"
        else:
            sched = (
                f"milestones={sorted(self._milestones)} "
                f"(frac={self.epoch_frac}, max_epochs={me})"
            )
        print(
            f"[eccv_val] callback ready: CE every epoch; STEP {sched}; "
            f"mid_n={len(self._mid_ids or [])} (fixed), "
            f"full_n={len(sample_ids) if self.max_samples <= 0 else min(len(sample_ids), self.max_samples)}, "
            f"gen_batch={self.gen_batch_size}",
            flush=True,
        )
        print(
            f"[eccv_val] mid ids: {(self._mid_ids or [])[: min(8, len(self._mid_ids or []))]}",
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
        if not self._should_run_official(trainer):
            return

        sample_ids, mode = self._ids_for_eval(trainer)
        if not sample_ids:
            print("[eccv_val] no official val sample ids; skip", flush=True)
            return

        work = self.work_dir or (
            Path(trainer.default_root_dir)
            / "metrics"
            / f"official_val_{mode}_epoch{int(trainer.current_epoch)+1:03d}"
            f"_step{int(trainer.global_step):06d}"
        )
        work.mkdir(parents=True, exist_ok=True)
        gt_dir, pred_dir, ok_ids = prepare_gt_pred_dirs(
            self.dataset_root, sample_ids, work
        )
        if not ok_ids:
            print("[eccv_val] no GT STEPs found; skip", flush=True)
            return

        print(
            f"[eccv_val] generating {len(ok_ids)} STEPs ({mode}) "
            f"batch={self.gen_batch_size} "
            f"(epoch={int(trainer.current_epoch)+1}/{self.max_epochs}, "
            f"step={trainer.global_step}) → {work}",
            flush=True,
        )
        was_training = pl_module.training
        device = next(pl_module.parameters()).device
        pl_module.eval()
        gen_log: list[dict[str, Any]] = []
        try:
            surface_fsq, edge_fsq = self._ensure_fsq(device)
            gen_log = generate_pred_steps_batched(
                transformer=pl_module,
                surface_fsq=surface_fsq,
                edge_fsq=edge_fsq,
                device=device,
                dataset_root=self.dataset_root,
                sample_ids=ok_ids,
                pred_dir=pred_dir,
                complexity=self.complexity,
                temperature=self.temperature,
                top_p=self.top_p,
                gen_batch_size=self.gen_batch_size,
            )
            for i, status in enumerate(gen_log):
                if (i + 1) % 10 == 0 or not status.get("ok") or i == 0:
                    print(
                        f"[eccv_val] [{i+1}/{len(ok_ids)}] {status.get('sample_id')}: "
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
