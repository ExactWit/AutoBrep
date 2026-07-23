"""AR generate with continuous prepend_embeds + KV cache.

x_transformers' AutoregressiveWrapper.generate() forwards ``prepend_embeds`` on
*every* decode step, which is unsafe with ``cache_kv=True``. Correct policy:

1. Prepend continuous condition only on the **first** forward.
2. Keep the full KV cache afterward (do not left-truncate; condition sits at head).
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from x_transformers.autoregressive_wrapper import top_p as default_top_p


@torch.inference_mode()
def generate_with_prepend_kv_cache(
    ar_wrapper,
    prompts: Tensor,
    *,
    seq_len: int,
    prepend_embeds: Optional[Tensor] = None,
    eos_token: Optional[int] = None,
    temperature: float = 1.0,
    filter_logits_fn: Optional[Callable] = None,
    filter_kwargs: Optional[dict] = None,
    pad_value: Optional[int] = None,
) -> Tensor:
    """
    Sample up to ``seq_len`` new tokens after ``prompts``.

    Returns only the newly generated tokens (same contract as
    ``AutoregressiveWrapper.generate``).

    When ``prepend_embeds`` is set we keep the full KV cache (no left-truncation):
    condition K/V sit at the head and must not be slid away. Rotary +
    ``can_cache_kv_outside_max_seq_len`` makes this safe for AutoBrep.
    """
    if filter_logits_fn is None:
        filter_logits_fn = default_top_p
    if filter_kwargs is None:
        filter_kwargs = {}
    if pad_value is None:
        pad_value = getattr(ar_wrapper, "pad_value", 0)

    net = ar_wrapper.net
    max_seq_len = ar_wrapper.max_seq_len
    greedy = temperature == 0.0
    has_prepend = prepend_embeds is not None

    out = prompts
    prompt_len = prompts.shape[-1]
    cache = None
    can_cache = bool(getattr(net, "can_cache_kv", True))

    was_training = ar_wrapper.training
    ar_wrapper.eval()
    try:
        for step in range(seq_len):
            # With continuous prepend, never left-truncate tokens/cache (would drop
            # condition). Without prepend, match stock sliding window.
            if has_prepend:
                x = out
            else:
                x = out[:, -max_seq_len:]
                if cache is not None:
                    for inter in cache.attn_intermediates:
                        if inter.layer_type == "a":
                            inter.cached_kv = [
                                t[..., -(max_seq_len - 1) :, :] for t in inter.cached_kv
                            ]

            # Prepend continuous condition only on the first forward.
            step_kwargs = {}
            if has_prepend and step == 0:
                step_kwargs["prepend_embeds"] = prepend_embeds

            logits, new_cache = net(
                x,
                return_intermediates=True,
                cache=cache,
                **step_kwargs,
            )

            if can_cache:
                cache = new_cache

            logits = logits[:, -1]

            if greedy:
                sample = logits.argmax(dim=-1, keepdim=True)
            else:
                filtered = filter_logits_fn(logits, **filter_kwargs)
                probs = F.softmax(filtered / temperature, dim=-1)
                sample = torch.multinomial(probs, 1)

            out = torch.cat((out, sample), dim=-1)

            if eos_token is None:
                continue

            is_eos = out == eos_token
            if is_eos.any(dim=-1).all():
                shifted = F.pad(is_eos, (1, -1))
                mask = shifted.float().cumsum(dim=-1) >= 1
                out = out.masked_fill(mask, pad_value)
                break
    finally:
        ar_wrapper.train(was_training)

    return out[:, prompt_len:]
