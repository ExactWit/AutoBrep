#!/usr/bin/env python3
"""Autocomplete showcase: several ABC / JSON condition cases → STEP gallery."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from pytorch_lightning import seed_everything

from autobrep.inference.autocomplete_prompt import (
    AutocompleteVocab,
    build_geom_tokens_from_arrays,
    condition_to_jsonable,
    find_abc_row_by_stem,
    load_condition_from_abc_row,
    load_condition_from_json,
    pick_adjacent_face_pair,
)
from autobrep.inference.brepgen_brep_builder import AutoBrepBuilder
from autobrep.inference.inference_common import reconstruct_compound
from autobrep.models.autoregressive import ARGenCheckpointPaths, AutoRegressiveSampler
from autobrep.models.vaes import EdgeFSQVAE, SurfaceFSQVAE
from autobrep.utils import DotDict, generate_random_string
from occwl.io import save_step as save_step_func

DEFAULT_WEIGHTS = "/data/hdd/outputs/AutoBrep"
DEFAULT_DATA = "/data/hdd/datasets/ABC-1M"
DEFAULT_STEM = "00007410_d681bb73885e41b39c681d22_step_047_0074"


def build_config(complexity: str = "medium") -> DotDict:
    return DotDict(
        {
            "seed": {"use_seed": True, "seed_value": 42},
            "debug_mode": False,
            "hyper_parameters": {
                "batch_size": 1,
                "complexity": complexity,
                "temperature": 1.0,
                "sample_method": {"top_p_threshold": 0.9},
                "vertex_threshold": 0.002,
                "sewing_tolerance": 0.002,
                "z_threshold": 0.0,
            },
        }
    )


def run_case(
    *,
    model: AutoRegressiveSampler,
    surf_enc,
    edge_enc,
    vocab: AutocompleteVocab,
    config: DotDict,
    case_dir: Path,
    case_id: str,
    cond: dict,
    device: torch.device,
) -> dict:
    case_dir.mkdir(parents=True, exist_ok=True)
    geom, remapped = build_geom_tokens_from_arrays(
        cond["face_pos"],
        cond["face_ncs"],
        cond["edge_pos"],
        cond["edge_ncs"],
        cond["face_edge_adj"],
        cond["user_face_indices"],
        surface_fsq=surf_enc,
        edge_fsq=edge_enc,
        device=device,
        vocab=vocab,
    )
    # export condition JSON for inspection
    payload = condition_to_jsonable(
        cond["face_pos"],
        cond["face_ncs"],
        cond["edge_pos"],
        cond["edge_ncs"],
        cond["face_edge_adj"],
        cond["user_face_indices"],
    )
    payload["meta"] = {
        **(cond.get("meta") or {}),
        "case_id": case_id,
        "geom_token_len": len(geom),
        "remapped_face_ids": remapped,
    }
    (case_dir / "condition.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )

    t0 = time.time()
    samples = model.sample_tokens(
        config=config, batch_size=1, geom_prompt_tokens=geom
    )
    gen_s = time.time() - t0

    decoded = model.decode_tokens(samples.detach().cpu().numpy())
    decode_fallback = False
    if not decoded:
        from autobrep.data.token_mapping import MMTokenIndex

        synth = np.array(
            [
                MMTokenIndex.BOS.value,
                *geom,
                MMTokenIndex.BOC.value,
                MMTokenIndex.EOC.value,
                MMTokenIndex.EOS.value,
            ],
            dtype=np.int64,
        )
        decoded = model.decode_tokens(synth[None, :])
        decode_fallback = bool(decoded)

    summary = {
        "case_id": case_id,
        "face_ids": cond["user_face_indices"],
        "geom_token_len": len(geom),
        "prompt_len": 4 + len(geom) + 1,
        "gen_seconds": round(gen_s, 2),
        "decode_ok": bool(decoded),
        "decode_fallback_geom_only": decode_fallback,
        "n_step": 0,
        "n_faces": None,
        "error": None,
        "meta": cond.get("meta"),
        "files": ["condition.json"],
    }
    if not decoded:
        summary["error"] = "decode_failed"
        (case_dir / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        return summary

    cad = model.convert_to_cad_data(decoded)[0]
    summary["n_faces"] = int(cad.face_pos_cad.shape[0])
    builders = [
        AutoBrepBuilder(
            device=model.device,
            z_threshold=0.0,
            vertex_threshold=0.005,
            sewing_tolerance=0.01,
        )
    ]
    stem = generate_random_string(12)

    def _try_rebuild(cad_data, tag: str) -> bool:
        try:
            result = reconstruct_compound(cad_data, builders)
            if result is None:
                return False
            step_name = f"{stem}_{tag}.step" if tag else f"{stem}.step"
            save_step_func([result], case_dir / step_name)
            summary["n_step"] = 1
            summary["step"] = step_name
            summary["files"].append(step_name)
            summary["error"] = None
            summary["rebuild_tag"] = tag or "full"
            return True
        except Exception as exc:  # noqa: BLE001
            summary["error"] = str(exc)
            return False

    ok = _try_rebuild(cad, "")
    if not ok:
        # Fall back: sew condition geometry only (official AR is zero-shot on BOGEOM)
        from autobrep.data.token_mapping import MMTokenIndex

        synth = np.array(
            [
                MMTokenIndex.BOS.value,
                *geom,
                MMTokenIndex.BOC.value,
                MMTokenIndex.EOC.value,
                MMTokenIndex.EOS.value,
            ],
            dtype=np.int64,
        )
        geom_decoded = model.decode_tokens(synth[None, :])
        if geom_decoded:
            geom_cad = model.convert_to_cad_data(geom_decoded)[0]
            summary["n_faces_geom_only"] = int(geom_cad.face_pos_cad.shape[0])
            if _try_rebuild(geom_cad, "geom_only"):
                summary["decode_fallback_geom_only"] = True
            else:
                summary["error"] = summary.get("error") or "rebuild_failed"
        else:
            summary["error"] = summary.get("error") or "rebuild_failed"

    (case_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (case_dir / f"{stem}.json").write_text(
        json.dumps(
            {
                "sample_id": stem,
                "autocomplete": True,
                **{k: summary[k] for k in summary if k != "files"},
                "files": summary["files"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return summary


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--output-dir",
        default="/data/hdd/outputs/autobrep_demo_showcase/autocomplete",
    )
    p.add_argument("--weight-folder", default=DEFAULT_WEIGHTS)
    p.add_argument("--data-dir", default=DEFAULT_DATA)
    p.add_argument("--stem", default=DEFAULT_STEM)
    p.add_argument("--gpu", default="0")
    args = p.parse_args()

    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda")
    wf = Path(args.weight_folder)
    model = AutoRegressiveSampler(
        checkpoint_paths=ARGenCheckpointPaths.from_folder(str(wf)),
        device=device,
    )
    surf_enc = (
        SurfaceFSQVAE.load_from_checkpoint(
            str(wf / "surf-fsq.ckpt"), map_location="cpu", use_dcae=True
        )
        .to(device)
        .eval()
    )
    edge_enc = (
        EdgeFSQVAE.load_from_checkpoint(
            str(wf / "edge-fsq.ckpt"), map_location="cpu", use_dcae=True
        )
        .to(device)
        .eval()
    )
    vocab = AutocompleteVocab(
        bit=int(model.transformer.hparams.bit),
        max_face=int(model.transformer.hparams.max_face),
        surf_codebook_size=int(model.transformer.hparams.surf_codebook_size),
        edge_codebook_size=int(model.transformer.hparams.edge_codebook_size),
    )
    seed_everything(42, workers=True)
    config = build_config("medium")

    row = find_abc_row_by_stem(args.data_dir, args.stem, split="train", max_files=5)
    pair = pick_adjacent_face_pair(
        __import__("autobrep.data.serialize", fromlist=["deserialize_array"])
        .deserialize_array(row["face_edge_incidence"])
        .astype(bool)
    )

    case_specs = []
    # A: adjacent pair
    case_specs.append(
        (
            "A_adjacent_pair",
            load_condition_from_abc_row(row, face_ids=pair),
        )
    )
    # B: constraint faces
    case_specs.append(
        (
            "B_constraint",
            load_condition_from_abc_row(row, mode="constraint"),
        )
    )
    # C: more faces (4)
    case_specs.append(
        (
            "C_four_faces",
            load_condition_from_abc_row(row, num_faces=4, mode="random"),
        )
    )
    # D: JSON roundtrip of A
    a_cond = case_specs[0][1]
    json_path = out / "case_A_condition.json"
    payload = condition_to_jsonable(
        a_cond["face_pos"],
        a_cond["face_ncs"],
        a_cond["edge_pos"],
        a_cond["edge_ncs"],
        a_cond["face_edge_adj"],
        a_cond["user_face_indices"],
    )
    json_path.write_text(json.dumps(payload), encoding="utf-8")
    case_specs.append(("D_json_from_A", load_condition_from_json(json_path)))

    results = []
    for i, (case_id, cond) in enumerate(case_specs):
        case_dir = out / f"{i:02d}_{case_id}"
        print(
            f"[ac-demo] === {case_id} faces={cond['user_face_indices']} ===",
            flush=True,
        )
        summary = run_case(
            model=model,
            surf_enc=surf_enc,
            edge_enc=edge_enc,
            vocab=vocab,
            config=config,
            case_dir=case_dir,
            case_id=case_id,
            cond=cond,
            device=device,
        )
        results.append(summary)
        print(
            f"[ac-demo] {case_id}: out_faces={summary.get('n_faces')} "
            f"step={summary.get('n_step')} err={summary.get('error')} "
            f"t={summary.get('gen_seconds')}s",
            flush=True,
        )

    lines = [
        "# AutoBrep Autocomplete 展示",
        "",
        f"- stem: `{args.stem}`",
        f"- weights: `{args.weight_folder}`",
        "",
        "| case | 条件面 | 生成面数 | STEP | 备注 |",
        "|------|--------|----------|------|------|",
    ]
    for i, s in enumerate(results):
        lines.append(
            f"| [{s['case_id']}]({i:02d}_{s['case_id']}/) | {s.get('face_ids')} | "
            f"{s.get('n_faces')} | {'✓' if s.get('n_step') else '—'} | "
            f"{s.get('error') or ''} |"
        )
    lines.append("")
    (out / "index.md").write_text("\n".join(lines), encoding="utf-8")
    (out / "gallery.json").write_text(
        json.dumps({"stem": args.stem, "results": results}, indent=2),
        encoding="utf-8",
    )
    print(f"[ac-demo] wrote {out}/index.md", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
