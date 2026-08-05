"""Prefix-LM attention masks for condition + AR token joint stacks."""

from __future__ import annotations

import torch


def build_prefix_lm_attn_mask(
    prefix_valid: torch.Tensor,
    seq_len: int,
) -> torch.Tensor:
    """
    Build a boolean attention mask (True = allow) for prefix-LM.

    Layout: ``[condition prefix | CAD tokens]`` with lengths ``P`` and ``S``.

    - Condition queries may only attend to **valid condition** keys (bidirectional).
    - CAD queries may attend to all valid condition keys and CAD keys with
      standard causal masking (position ``P+i`` sees ``P..P+i``).
    - Non-condition information therefore cannot flow into condition tokens.

    Args:
        prefix_valid: ``(B, P)`` bool, True = real condition token (not pad).
        seq_len: discrete CAD token length ``S`` (including BOS…EOS pad length).

    Returns:
        ``(B, P+S, P+S)`` bool mask.
    """
    if prefix_valid.ndim != 2:
        raise ValueError(f"prefix_valid must be (B, P), got {tuple(prefix_valid.shape)}")
    if seq_len < 1:
        raise ValueError(f"seq_len must be >= 1, got {seq_len}")

    b, p = prefix_valid.shape
    s = int(seq_len)
    device = prefix_valid.device
    t = p + s
    allow = torch.zeros(b, t, t, dtype=torch.bool, device=device)

    # Condition ↔ condition (bidirectional among valid tokens).
    # allow[:, i, j] = valid_i & valid_j for i,j < P
    allow[:, :p, :p] = prefix_valid.unsqueeze(2) & prefix_valid.unsqueeze(1)

    # CAD → condition (all valid prefix keys).
    allow[:, p:, :p] = prefix_valid.unsqueeze(1).expand(b, s, p)

    # CAD → CAD (causal).
    causal = torch.ones(s, s, dtype=torch.bool, device=device).tril()
    allow[:, p:, p:] = causal

    return allow
