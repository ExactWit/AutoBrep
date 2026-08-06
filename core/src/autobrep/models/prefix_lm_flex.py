"""Prefix-LM attention via PyTorch flex_attention (block-sparse, no dense T² mask).

Installs a small patch on ``x_transformers.Attend.flash_attn`` so that while a
``prefix_lm_flex_attention`` context is active, SDPA with a materialised bool
mask is skipped and ``flex_attention`` + ``BlockMask`` is used instead.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Iterator, Optional

import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

_CTX: ContextVar[Optional["PrefixLMFlexState"]] = ContextVar(
    "autobrep_prefix_lm_flex", default=None
)
_PATCHED = False
_ORIG_FLASH_ATTN = None
_COMPILED_FLEX = None


def _get_compiled_flex():
    """Lazy ``torch.compile(flex_attention)`` — uncompiled path materializes T² scores."""
    global _COMPILED_FLEX
    if _COMPILED_FLEX is None:
        _COMPILED_FLEX = torch.compile(flex_attention)
    return _COMPILED_FLEX


@dataclass
class PrefixLMFlexState:
    block_mask: object
    prefix_len: int
    seq_len: int


def _build_block_mask(
    prefix_valid: torch.Tensor,
    seq_len: int,
    cad_valid: torch.Tensor | None = None,
):
    """
    Args:
        prefix_valid: (B, P) bool
        seq_len: CAD length S (packed batch max)
        cad_valid: optional (B, S) bool; defaults to all-True
    """
    if prefix_valid.ndim != 2:
        raise ValueError(f"prefix_valid must be (B, P), got {tuple(prefix_valid.shape)}")
    bsz, p = prefix_valid.shape
    s = int(seq_len)
    if s < 1:
        raise ValueError(f"seq_len must be >= 1, got {s}")
    t = p + s
    device = prefix_valid.device
    valid_p = prefix_valid.to(device=device, dtype=torch.bool)
    if cad_valid is None:
        valid_c = torch.ones(bsz, s, dtype=torch.bool, device=device)
    else:
        if cad_valid.shape != (bsz, s):
            raise ValueError(
                f"cad_valid must be {(bsz, s)}, got {tuple(cad_valid.shape)}"
            )
        valid_c = cad_valid.to(device=device, dtype=torch.bool)

    # Capture in closure for mask_mod (flex compiles / traces this).
    def mask_mod(b, h, q_idx, kv_idx):
        del h  # unused; kept for flex signature
        is_pq = q_idx < p
        is_pk = kv_idx < p
        q_pi = torch.where(is_pq, q_idx, torch.zeros_like(q_idx))
        kv_pi = torch.where(is_pk, kv_idx, torch.zeros_like(kv_idx))
        q_ci = torch.where(~is_pq, q_idx - p, torch.zeros_like(q_idx))
        kv_ci = torch.where(~is_pk, kv_idx - p, torch.zeros_like(kv_idx))

        q_ok = torch.where(is_pq, valid_p[b, q_pi], valid_c[b, q_ci])
        kv_ok = torch.where(is_pk, valid_p[b, kv_pi], valid_c[b, kv_ci])

        both_prefix = is_pq & is_pk & q_ok & kv_ok
        cad_to_prefix = (~is_pq) & is_pk & q_ok & kv_ok
        cad_causal = (~is_pq) & (~is_pk) & q_ok & kv_ok & (kv_idx <= q_idx)
        return both_prefix | cad_to_prefix | cad_causal

    return create_block_mask(
        mask_mod,
        B=bsz,
        H=None,
        Q_LEN=t,
        KV_LEN=t,
        device=device,
    )


def ensure_attend_patched() -> None:
    """Idempotently patch x_transformers Attend.flash_attn for flex routing."""
    global _PATCHED, _ORIG_FLASH_ATTN
    if _PATCHED:
        return
    from x_transformers.attend import Attend

    _ORIG_FLASH_ATTN = Attend.flash_attn

    def _flex_flash_attn(
        self,
        q,
        k,
        v,
        mask=None,
        attn_bias=None,
        flash_pack_seq_kwargs=None,
    ):
        ctx = _CTX.get()
        if ctx is not None:
            # Match Attend.flash_attn contract: (out, Intermediates)
            from x_transformers.attend import Intermediates

            # q/k/v: (batch, heads, seq, dim)
            out = _get_compiled_flex()(q, k, v, block_mask=ctx.block_mask)
            if self.training and float(self.dropout) > 0:
                out = F.dropout(out, p=float(self.dropout))
            return out, Intermediates()
        return _ORIG_FLASH_ATTN(
            self,
            q,
            k,
            v,
            mask=mask,
            attn_bias=attn_bias,
            flash_pack_seq_kwargs=flash_pack_seq_kwargs,
        )

    Attend.flash_attn = _flex_flash_attn  # type: ignore[method-assign]
    _PATCHED = True


@contextmanager
def prefix_lm_flex_attention(
    prefix_valid: torch.Tensor,
    seq_len: int,
    cad_valid: torch.Tensor | None = None,
) -> Iterator[PrefixLMFlexState]:
    """Run AR forward/generate step-0 under flex prefix-LM masking."""
    ensure_attend_patched()
    block_mask = _build_block_mask(prefix_valid, seq_len, cad_valid=cad_valid)
    state = PrefixLMFlexState(
        block_mask=block_mask,
        prefix_len=int(prefix_valid.shape[1]),
        seq_len=int(seq_len),
    )
    token = _CTX.set(state)
    try:
        yield state
    finally:
        _CTX.reset(token)
