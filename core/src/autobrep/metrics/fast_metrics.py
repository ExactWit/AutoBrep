"""Level-1 fast metrics: PPL / token acc / light topology compliance (no OCC)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as F

from autobrep.data.token_mapping import MMTokenIndex

# Special / structural tokens (not FSQ geom codes)
_SPECIAL_MAX = int(MMTokenIndex.DUMMYID.value)
_COMPLEXITY = {
    int(MMTokenIndex.GEN_EASY.value),
    int(MMTokenIndex.GEN_MID.value),
    int(MMTokenIndex.GEN_HARD.value),
    int(MMTokenIndex.GEN_UNCOND.value),
}
_TOPO_MARKERS = {
    int(MMTokenIndex.BOC.value),
    int(MMTokenIndex.EOC.value),
    int(MMTokenIndex.BOL.value),
    int(MMTokenIndex.EOL.value),
    int(MMTokenIndex.BOF.value),
    int(MMTokenIndex.EOF.value),
    int(MMTokenIndex.BOM.value),
    int(MMTokenIndex.EOM.value),
}


@dataclass
class FastMetricResult:
    ppl: float
    token_acc: float
    geom_acc: float
    topo_acc: float
    complexity_acc: float
    topo_compliance_rate: float
    n_tokens: int
    n_sequences: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def log_dict(self, prefix: str = "val/fast") -> dict[str, float]:
        return {
            f"{prefix}/ppl": self.ppl,
            f"{prefix}/token_acc": self.token_acc,
            f"{prefix}/geom_acc": self.geom_acc,
            f"{prefix}/topo_acc": self.topo_acc,
            f"{prefix}/complexity_acc": self.complexity_acc,
            f"{prefix}/topo_compliance": self.topo_compliance_rate,
        }


def bucket_token_mask(targets: torch.Tensor) -> dict[str, torch.Tensor]:
    """Boolean masks over flat target tokens (ignore_index=-1 already filtered)."""
    t = targets.long()
    special = t <= _SPECIAL_MAX
    complexity = torch.zeros_like(t, dtype=torch.bool)
    topo = torch.zeros_like(t, dtype=torch.bool)
    for v in _COMPLEXITY:
        complexity |= t == v
    for v in _TOPO_MARKERS:
        topo |= t == v
    geom = (~special) | (t > _SPECIAL_MAX)
    # FSQ / bbox-style codes live above special range
    geom = t > _SPECIAL_MAX
    return {
        "geom": geom,
        "topo": topo,
        "complexity": complexity,
        "special": special,
    }


def _safe_acc(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    if mask is None or not bool(mask.any()):
        return float("nan")
    correct = (pred[mask] == target[mask]).float()
    return float(correct.mean().item())


def light_topo_compliance(seq: torch.Tensor, *, max_face: int = 200) -> float:
    """
    Rule-based topology sanity on a 1D token sequence (no OCC).

    Checks:
    - BOF/EOF nesting balanced within BOC/EOC
    - BOL/EOL nesting balanced
    - BOM/EOM at most one pair
    - Face count implied by BOF does not exceed max_face
    Returns 1.0 if compliant else 0.0.
    """
    if seq.ndim != 1:
        seq = seq.reshape(-1)
    tokens = [int(x) for x in seq.tolist() if int(x) >= 0]
    if not tokens:
        return 0.0

    bof = int(MMTokenIndex.BOF.value)
    eof = int(MMTokenIndex.EOF.value)
    bol = int(MMTokenIndex.BOL.value)
    eol = int(MMTokenIndex.EOL.value)
    boc = int(MMTokenIndex.BOC.value)
    eoc = int(MMTokenIndex.EOC.value)
    bom = int(MMTokenIndex.BOM.value)
    eom = int(MMTokenIndex.EOM.value)

    face_depth = 0
    level_depth = 0
    cad_depth = 0
    meta_depth = 0
    n_faces = 0
    for t in tokens:
        if t == boc:
            cad_depth += 1
        elif t == eoc:
            cad_depth -= 1
            if cad_depth < 0:
                return 0.0
        elif t == bol:
            level_depth += 1
        elif t == eol:
            level_depth -= 1
            if level_depth < 0:
                return 0.0
        elif t == bof:
            face_depth += 1
            n_faces += 1
            if n_faces > max_face:
                return 0.0
        elif t == eof:
            face_depth -= 1
            if face_depth < 0:
                return 0.0
        elif t == bom:
            meta_depth += 1
            if meta_depth > 1:
                return 0.0
        elif t == eom:
            meta_depth -= 1
            if meta_depth < 0:
                return 0.0

    if face_depth != 0 or level_depth != 0 or cad_depth != 0 or meta_depth != 0:
        return 0.0
    return 1.0


def compute_fast_metrics_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    ignore_index: int = -1,
    cond_mask: torch.Tensor | None = None,
    full_sequences: torch.Tensor | None = None,
    max_face: int = 200,
) -> FastMetricResult:
    """
    Args:
        logits: (B, T, V) predicting targets of shape (B, T)
        targets: (B, T) next-token labels
        cond_mask: (B, T) True = ignore (same convention as XTransformer)
        full_sequences: optional (B, L) token ids for light topo check
    """
    if logits.ndim != 3:
        raise ValueError(f"logits must be (B,T,V), got {tuple(logits.shape)}")
    pred = logits.argmax(dim=-1)
    valid = targets != ignore_index
    if cond_mask is not None:
        # cond_mask True = ignore
        if cond_mask.shape != targets.shape:
            raise ValueError("cond_mask shape must match targets")
        valid = valid & (~cond_mask.to(device=valid.device, dtype=torch.bool))

    flat_pred = pred[valid]
    flat_tgt = targets[valid]
    n = int(flat_tgt.numel())
    if n == 0:
        return FastMetricResult(
            ppl=float("nan"),
            token_acc=float("nan"),
            geom_acc=float("nan"),
            topo_acc=float("nan"),
            complexity_acc=float("nan"),
            topo_compliance_rate=float("nan"),
            n_tokens=0,
            n_sequences=int(targets.size(0)),
        )

    ce = F.cross_entropy(logits[valid], flat_tgt, reduction="mean")
    ppl = float(torch.exp(ce.detach()).item())
    token_acc = float((flat_pred == flat_tgt).float().mean().item())
    buckets = bucket_token_mask(flat_tgt)
    geom_acc = _safe_acc(flat_pred, flat_tgt, buckets["geom"])
    topo_acc = _safe_acc(flat_pred, flat_tgt, buckets["topo"])
    complexity_acc = _safe_acc(flat_pred, flat_tgt, buckets["complexity"])

    if full_sequences is None:
        topo_rate = float("nan")
        n_seq = int(targets.size(0))
    else:
        rates = [
            light_topo_compliance(full_sequences[i], max_face=max_face)
            for i in range(full_sequences.size(0))
        ]
        topo_rate = float(sum(rates) / max(len(rates), 1))
        n_seq = len(rates)

    return FastMetricResult(
        ppl=ppl,
        token_acc=token_acc,
        geom_acc=geom_acc if geom_acc == geom_acc else float("nan"),
        topo_acc=topo_acc if topo_acc == topo_acc else float("nan"),
        complexity_acc=complexity_acc if complexity_acc == complexity_acc else float("nan"),
        topo_compliance_rate=topo_rate,
        n_tokens=n,
        n_sequences=n_seq,
    )
