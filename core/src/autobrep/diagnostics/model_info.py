"""Pretty-print / dump model module tree, params, FLOPs, and wiring."""

from __future__ import annotations

import json
import sys
from collections import OrderedDict
from typing import Any, Callable, Optional, TextIO

import torch
import torch.nn as nn


def _n_params(module: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in module.parameters())
    train = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, train


def _fmt_num(n: int | float) -> str:
    n = float(n)
    abs_n = abs(n)
    if abs_n >= 1e9:
        return f"{n/1e9:.3f}B"
    if abs_n >= 1e6:
        return f"{n/1e6:.3f}M"
    if abs_n >= 1e3:
        return f"{n/1e3:.2f}K"
    return f"{n:.0f}"


def collect_module_tree(
    root: nn.Module,
    *,
    max_depth: int = 3,
    prefix: str = "",
) -> list[dict[str, Any]]:
    """Top-level children + limited nested named_children (not every leaf)."""
    rows: list[dict[str, Any]] = []

    def walk(mod: nn.Module, name: str, depth: int) -> None:
        total, train = _n_params(mod)
        rows.append(
            {
                "name": name or type(mod).__name__,
                "type": type(mod).__name__,
                "depth": depth,
                "params_total": total,
                "params_trainable": train,
                "num_children": len(list(mod.children())),
            }
        )
        if depth >= max_depth:
            return
        for child_name, child in mod.named_children():
            path = f"{name}.{child_name}" if name else child_name
            walk(child, path, depth + 1)

    walk(root, prefix or type(root).__name__, 0)
    return rows


def connection_graph_autobrep_view() -> dict[str, Any]:
    """Static dataflow for AutoBrepViewModel (condition SFT)."""
    return {
        "nodes": [
            {"id": "images", "label": "images (B,3,3,H,W) renders"},
            {"id": "dxf", "label": "TechDraw DXF prim_* tensors"},
            {"id": "view_encoder.backbone", "label": "ResNet-18 (shared)"},
            {"id": "view_encoder.techdraw_encoder", "label": "TechDrawSetEncoder"},
            {"id": "view_encoder.cross_attn", "label": "latent cross-attn"},
            {"id": "view_encoder.proj", "label": "proj → dim=2048"},
            {"id": "prepend", "label": "prepend_embeds (B,M+2,D)"},
            {"id": "surf_vae", "label": "SurfaceFSQVAE (frozen)"},
            {"id": "edge_vae", "label": "EdgeFSQVAE (frozen)"},
            {"id": "face_ncs", "label": "face_ncs / edge_ncs"},
            {"id": "fsq_codes", "label": "FSQ geom tokens"},
            {"id": "seq", "label": "AR token seq (BOS+meta+CAD)"},
            {"id": "cad_gpt", "label": "XTransformer AR (frozen)"},
            {"id": "ce", "label": "NTP CE loss"},
        ],
        "edges": [
            ["images", "view_encoder.backbone"],
            ["view_encoder.backbone", "view_encoder.cross_attn"],
            ["dxf", "view_encoder.techdraw_encoder"],
            ["view_encoder.techdraw_encoder", "view_encoder.cross_attn"],
            ["view_encoder.cross_attn", "view_encoder.proj"],
            ["view_encoder.proj", "prepend"],
            ["face_ncs", "surf_vae"],
            ["face_ncs", "edge_vae"],
            ["surf_vae", "fsq_codes"],
            ["edge_vae", "fsq_codes"],
            ["seq", "fsq_codes"],
            ["fsq_codes", "cad_gpt"],
            ["prepend", "cad_gpt"],
            ["cad_gpt", "ce"],
        ],
        "train_path": "ce → (frozen cad_gpt) → prepend → view_encoder only",
        "mermaid": "\n".join(
            [
                "flowchart LR",
                "  IMG[3 renders] --> RN[ResNet18]",
                "  DXF[DXF prims] --> TD[TechDrawSetEnc]",
                "  RN --> XA[cross-attn latents]",
                "  TD --> XA",
                "  XA --> PRE[prepend_embeds]",
                "  FNC[face/edge NCS] --> FSQ[FSQ VAE frozen]",
                "  FSQ --> TOK[AR tokens]",
                "  SEQ[seq + meta] --> TOK",
                "  PRE --> AR[XTransformer frozen]",
                "  TOK --> AR",
                "  AR --> CE[NTP CE]",
                "  CE -.grad.-> PRE",
                "  PRE -.grad.-> XA",
            ]
        ),
    }


@torch.inference_mode()
def estimate_flops_module(
    module: nn.Module,
    inputs: tuple,
    *,
    backend: str = "auto",
) -> dict[str, Any]:
    """Profile FLOPs for ``module(*inputs)`` via thop or FlopCounterMode."""
    errors: list[str] = []

    if backend in ("auto", "thop"):
        try:
            from thop import profile

            flops, params = profile(module, inputs=inputs, verbose=False)
            return {
                "backend": "thop",
                "flops_total": int(flops),
                "flops_total_human": _fmt_num(flops),
                "params_reported": int(params),
            }
        except Exception as exc:  # noqa: BLE001
            errors.append(f"thop: {type(exc).__name__}: {exc}")
            if backend == "thop":
                return {"error": "; ".join(errors)}

    if backend in ("auto", "native"):
        try:
            from torch.utils.flop_counter import FlopCounterMode

            with FlopCounterMode(display=False) as fcm:
                module(*inputs)
            total = int(fcm.get_total_flops())
            top: list[tuple[str, int]] = []
            try:
                counts = fcm.get_flop_counts() or {}
                # counts may be nested {module: {op: n}}
                flat: dict[str, int] = {}
                for key, val in counts.items():
                    if isinstance(val, dict):
                        for op, n in val.items():
                            flat[f"{key}:{op}"] = int(n)
                    else:
                        flat[str(key)] = int(val)
                top = sorted(flat.items(), key=lambda kv: kv[1], reverse=True)[:12]
            except Exception:  # noqa: BLE001
                pass
            return {
                "backend": "FlopCounterMode",
                "flops_total": total,
                "flops_total_human": _fmt_num(total),
                "top_ops": top,
            }
        except Exception as exc:  # noqa: BLE001
            errors.append(f"native: {type(exc).__name__}: {exc}")

    return {"error": "; ".join(errors) if errors else "no backend"}


@torch.inference_mode()
def estimate_view_encoder_flops(
    view_encoder: nn.Module, *, device: torch.device
) -> dict[str, Any]:
    """
    Break down FLOPs:
      - ResNet backbone × 3 views (thop)
      - TechDraw set encoder + fusion (native / thop best-effort)
    """
    enc = view_encoder.to(device).eval()
    b, v, h, w = 1, int(getattr(enc, "num_image_views", 3)), 224, 224
    images = torch.randn(b, v, 3, h, w, device=device, dtype=torch.float32)
    prim_types = torch.zeros(b, 3, 128, device=device, dtype=torch.long)
    prim_linetypes = torch.zeros(b, 3, 128, device=device, dtype=torch.long)
    prim_geom = torch.randn(b, 3, 128, 12, device=device, dtype=torch.float32)
    prim_mask = torch.zeros(b, 3, 128, device=device, dtype=torch.bool)
    prim_mask[:, :, :32] = True

    out: dict[str, Any] = {"input": {"B": b, "V": v, "H": h, "W": w, "td_views": 3, "max_prims": 128}}

    # Backbone (single image) × V
    one = images[:, 0]
    bb = estimate_flops_module(enc.backbone, (one,))
    out["backbone_per_view"] = bb
    if "flops_total" in bb:
        out["backbone_all_views"] = {
            "flops_total": bb["flops_total"] * v,
            "flops_total_human": _fmt_num(bb["flops_total"] * v),
            "num_views": v,
        }

    # Full view_encoder (may fail on nested-tensor Transformer path)
    full = estimate_flops_module(
        enc, (images, prim_types, prim_linetypes, prim_geom, prim_mask)
    )
    out["full_view_encoder"] = full

    if "flops_total" in full:
        out["flops_total"] = full["flops_total"]
        out["flops_total_human"] = full["flops_total_human"]
        out["backend"] = full.get("backend")
    elif "flops_total" in bb:
        # Approximate: backbone×V + rough TechDraw (ignore if full failed)
        approx = bb["flops_total"] * v
        out["flops_total"] = approx
        out["flops_total_human"] = _fmt_num(approx)
        out["backend"] = "approx_backbone_x_views"
        out["approx_note"] = (
            "Full encoder FLOP counter failed; reported ResNet×V only. "
            f"full_error={full.get('error')}"
        )
    else:
        out["error"] = full.get("error") or bb.get("error")

    return out


def estimate_flops(
    module: nn.Module,
    example_fn: Callable[[], Any],
    *,
    device: torch.device,
) -> Optional[dict[str, Any]]:
    """Legacy wrapper kept for callers; prefer estimate_view_encoder_flops."""
    try:
        from torch.utils.flop_counter import FlopCounterMode
    except ImportError:
        return None

    module = module.to(device)
    was_training = module.training
    module.eval()
    try:
        with FlopCounterMode(display=False) as fcm:
            example_fn()
        total = int(fcm.get_total_flops())
        return {"flops_total": total, "flops_total_human": _fmt_num(total)}
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}
    finally:
        module.train(was_training)


def _view_encoder_example(model: nn.Module, device: torch.device):
    enc = model.view_encoder
    b, v, h, w = 1, 3, 224, 224
    images = torch.zeros(b, v, 3, h, w, device=device, dtype=torch.float32)
    prim_types = torch.zeros(b, 3, 128, device=device, dtype=torch.long)
    prim_linetypes = torch.zeros(b, 3, 128, device=device, dtype=torch.long)
    prim_geom = torch.zeros(b, 3, 128, 12, device=device, dtype=torch.float32)
    prim_mask = torch.zeros(b, 3, 128, device=device, dtype=torch.bool)
    prim_mask[:, :, :8] = True
    return enc(images, prim_types, prim_linetypes, prim_geom, prim_mask)


def build_model_info_report(
    model: nn.Module,
    *,
    device: Optional[torch.device] = None,
    max_depth: int = 2,
    profile_view_encoder_flops: bool = True,
) -> dict[str, Any]:
    device = device or next(model.parameters()).device
    total, train = _n_params(model)
    children: list[dict[str, Any]] = []
    for name, child in model.named_children():
        ct, ctr = _n_params(child)
        children.append(
            {
                "name": name,
                "type": type(child).__name__,
                "params_total": ct,
                "params_trainable": ctr,
                "params_total_human": _fmt_num(ct),
                "params_trainable_human": _fmt_num(ctr),
                "frozen": ctr == 0 and ct > 0,
            }
        )

    tree = collect_module_tree(model, max_depth=max_depth)
    graph = connection_graph_autobrep_view()

    flops_section: dict[str, Any] = {}
    if profile_view_encoder_flops and hasattr(model, "view_encoder"):
        flops_section["view_encoder"] = estimate_view_encoder_flops(
            model.view_encoder, device=device
        )
        flops_section["note"] = (
            "FLOPs for condition path (view_encoder). "
            "Frozen cad_gpt NTP + FSQ encode FLOPs are omitted "
            "(dominated by AR; not needed for trainable-head sizing)."
        )

    return OrderedDict(
        [
            ("model_class", type(model).__name__),
            (
                "params",
                {
                    "total": total,
                    "trainable": train,
                    "frozen": total - train,
                    "total_human": _fmt_num(total),
                    "trainable_human": _fmt_num(train),
                    "trainable_ratio": (train / total) if total else 0.0,
                },
            ),
            ("top_modules", children),
            ("module_tree", tree),
            ("connections", graph),
            ("flops", flops_section),
        ]
    )


def format_model_info_text(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append(f"MODEL INFO — {report.get('model_class')}")
    lines.append("=" * 72)
    p = report["params"]
    lines.append(
        f"Params: total={p['total_human']} ({p['total']:,})  "
        f"trainable={p['trainable_human']} ({p['trainable']:,})  "
        f"frozen={_fmt_num(p['frozen'])}  "
        f"trainable_ratio={p['trainable_ratio']*100:.3f}%"
    )
    lines.append("")
    lines.append("--- Top-level modules ---")
    lines.append(
        f"{'name':<18} {'type':<28} {'params':>12} {'trainable':>12} {'status':>8}"
    )
    for m in report["top_modules"]:
        status = "FROZEN" if m["frozen"] else ("TRAIN" if m["params_trainable"] else "empty")
        lines.append(
            f"{m['name']:<18} {m['type']:<28} "
            f"{m['params_total_human']:>12} {m['params_trainable_human']:>12} {status:>8}"
        )

    lines.append("")
    lines.append("--- Module tree (depth-limited) ---")
    for row in report["module_tree"]:
        indent = "  " * int(row["depth"])
        lines.append(
            f"{indent}{row['name']}  [{row['type']}]  "
            f"params={_fmt_num(row['params_total'])}  "
            f"train={_fmt_num(row['params_trainable'])}"
        )

    conn = report.get("connections") or {}
    lines.append("")
    lines.append("--- Connections (dataflow) ---")
    lines.append(f"train_path: {conn.get('train_path', '')}")
    for a, b in conn.get("edges") or []:
        lines.append(f"  {a}  →  {b}")
    if conn.get("mermaid"):
        lines.append("")
        lines.append("mermaid:")
        lines.append(conn["mermaid"])

    flops = report.get("flops") or {}
    lines.append("")
    lines.append("--- FLOPs ---")
    if flops.get("note"):
        lines.append(flops["note"])
    ve = flops.get("view_encoder")
    if isinstance(ve, dict):
        if "error" in ve and "flops_total" not in ve:
            lines.append(f"view_encoder FLOPs: ERROR {ve['error']}")
        else:
            lines.append(
                f"view_encoder FLOPs: {ve.get('flops_total_human')} "
                f"({ve.get('flops_total', 0):,})  backend={ve.get('backend')}"
            )
            if ve.get("approx_note"):
                lines.append(f"  note: {ve['approx_note']}")
            bb = ve.get("backbone_per_view") or {}
            if "flops_total_human" in bb:
                lines.append(
                    f"  ResNet per view: {bb['flops_total_human']}  "
                    f"×{ve.get('backbone_all_views', {}).get('num_views', 3)} → "
                    f"{ve.get('backbone_all_views', {}).get('flops_total_human', '?')}"
                )
            full = ve.get("full_view_encoder") or {}
            if full.get("error"):
                lines.append(f"  full encoder profile: {full['error']}")
            elif "flops_total_human" in full:
                lines.append(f"  full encoder: {full['flops_total_human']}")
    lines.append("=" * 72)
    return "\n".join(lines)


def log_model_info(
    model: nn.Module,
    *,
    out_path: Optional[str] = None,
    stream: TextIO = sys.stderr,
    max_depth: int = 2,
) -> dict[str, Any]:
    report = build_model_info_report(model, max_depth=max_depth)
    text = format_model_info_text(report)
    print(text, file=stream, flush=True)
    if out_path:
        payload = json.loads(json.dumps(report))  # ensure JSON-serializable
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"[model_info] wrote {out_path}", file=stream, flush=True)
    return report
