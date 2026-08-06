"""AR generate with continuous prepend_embeds + KV cache.

x_transformers applies rotary to ``cat(cached_kv, new_kv)``. Freqs must span the
full cached length, which requires ``input_not_include_cache=True`` so
``seq_pos_offset = cache.cache_length`` and positions are
``arange(new_len + cache_length)``.

Policy:
1. Step 0: forward with ``prepend_embeds`` (condition + prompt).
2. Later steps: feed only the newest token(s); do **not** re-pass prepend.
3. Always pass ``input_not_include_cache=True`` when using the KV cache.
4. Optional ``prefix_valid`` activates flex prefix-LM masking on step 0 only
   (``causal=False``); later steps use KV cache under default causal.
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
    prefix_valid: Optional[Tensor] = None,
    eos_token: Optional[int] = None,
    temperature: float = 1.0,
    filter_logits_fn: Optional[Callable] = None,
    filter_kwargs: Optional[dict] = None,
    pad_value: Optional[int] = None,
) -> Tensor:
    """
    Sample up to ``seq_len`` new tokens after ``prompts``.

    Returns only the newly generated tokens (same contract as
    ``AutoregressiveWrapper.generate``). Greedy outputs match
    ``cache_kv=False`` when ``prepend_embeds`` is used.
    """
    if filter_logits_fn is None:
        filter_logits_fn = default_top_p
    if filter_kwargs is None:
        filter_kwargs = {}
    if pad_value is None:
        pad_value = getattr(ar_wrapper, "pad_value", 0)

    net = ar_wrapper.net
    greedy = temperature == 0.0
    has_prepend = prepend_embeds is not None

    # No continuous prefix → stock path (already KV-cached).
    if not has_prepend:
        return ar_wrapper.generate(
            prompts=prompts,
            seq_len=seq_len,
            eos_token=eos_token,
            temperature=temperature,
            filter_logits_fn=filter_logits_fn,
            filter_kwargs=filter_kwargs,
            cache_kv=True,
        )

    out = prompts
    prompt_len = prompts.shape[-1]
    cache = None
    can_cache = bool(getattr(net, "can_cache_kv", True))

    was_training = ar_wrapper.training
    ar_wrapper.eval()
    try:
        for step in range(seq_len):
            if step == 0:
                x = out
                step_kwargs = {
                    "prepend_embeds": prepend_embeds,
                    "input_not_include_cache": True,
                }
                if prefix_valid is not None:
                    from autobrep.models.prefix_lm_flex import prefix_lm_flex_attention

                    # Flex mask covers prepend + prompt; disable default causal.
                    with prefix_lm_flex_attention(prefix_valid, seq_len=prompt_len):
                        step_kwargs["causal"] = False
                        logits, new_cache = net(
                            x,
                            return_intermediates=True,
                            cache=cache,
                            **step_kwargs,
                        )
                else:
                    logits, new_cache = net(
                        x,
                        return_intermediates=True,
                        cache=cache,
                        **step_kwargs,
                    )
            else:
                # Newest token only; cache already holds condition + past tokens.
                x = out[:, -1:]
                step_kwargs = {"input_not_include_cache": True}
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
