"""P1-B: inject per-layer cross-attn into a frozen x-transformers Decoder."""

from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn as nn


class LayerCrossAttn(nn.Module):
    """Pre-norm MultiheadAttention residual block (Q=decoder, K/V=encoder mem)."""

    def __init__(self, dim: int, n_heads: int = 8, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        # Zero-init out_proj so injection starts as identity
        nn.init.zeros_(self.attn.out_proj.weight)
        if self.attn.out_proj.bias is not None:
            nn.init.zeros_(self.attn.out_proj.bias)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        *,
        context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T, D) decoder branch output
            context: (B, L, D) encoder memory
            context_mask: (B, L) True = valid (will invert for key_padding_mask)
        """
        q = self.norm(x)
        key_pad = None
        if context_mask is not None:
            key_pad = ~context_mask.bool()
            if (~key_pad).any():
                # ensure at least one valid key per row
                empty = key_pad.all(dim=1)
                if empty.any():
                    key_pad = key_pad.clone()
                    key_pad[empty, 0] = False
            else:
                key_pad = None
        out, _ = self.attn(q, context, context, key_padding_mask=key_pad, need_weights=False)
        return x + out


class DecoderCrossAttnInjector(nn.Module):
    """
    Attach learnable cross-attn modules to each self-attention ('a') layer of an
    x-transformers ``AttentionLayers`` via forward hooks.

    Frozen backbone stays unchanged; only ``cross_layers`` are trained.
    Call ``set_context`` before each AR forward / generate step.
    """

    def __init__(
        self,
        attn_layers: nn.Module,
        *,
        dim: int,
        n_heads: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        # Hold without registering as submodule (backbone already under cad_gpt).
        object.__setattr__(self, "attn_layers", attn_layers)
        # Count self-attn layers
        layer_types = list(getattr(attn_layers, "layer_types", []))
        n_a = sum(1 for t in layer_types if t == "a")
        if n_a <= 0:
            # Fallback: depth attribute
            n_a = int(getattr(attn_layers, "depth", 16) or 16)
        self.cross_layers = nn.ModuleList(
            [LayerCrossAttn(dim, n_heads=n_heads, dropout=dropout) for _ in range(n_a)]
        )
        self._context: Optional[torch.Tensor] = None
        self._context_mask: Optional[torch.Tensor] = None
        self._hooks: list = []
        self._enabled = True
        self._install_hooks()

    def _install_hooks(self) -> None:
        self.remove_hooks()
        layers = getattr(self.attn_layers, "layers", None)
        layer_types = list(getattr(self.attn_layers, "layer_types", []))
        if layers is None or not layer_types:
            return
        a_idx = 0
        for i, layer_type in enumerate(layer_types):
            if layer_type != "a":
                continue
            if a_idx >= len(self.cross_layers):
                break
            # layers[i] = (norm, block, residual_fn)
            block = layers[i][1]
            cross = self.cross_layers[a_idx]

            def _make_hook(cross_mod: LayerCrossAttn) -> Callable:
                def _hook(_module, _inp, out):
                    if not self._enabled or self._context is None:
                        return out
                    ctx = self._context
                    mask = self._context_mask
                    if isinstance(out, tuple):
                        tensor = out[0]
                        # Align dtype/device
                        ctx_u = ctx.to(dtype=tensor.dtype, device=tensor.device)
                        updated = cross_mod(tensor, ctx_u, context_mask=mask)
                        return (updated, *out[1:])
                    ctx_u = ctx.to(dtype=out.dtype, device=out.device)
                    return cross_mod(out, ctx_u, context_mask=mask)

                return _hook

            self._hooks.append(block.register_forward_hook(_make_hook(cross)))
            a_idx += 1

    def remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def set_context(
        self,
        context: Optional[torch.Tensor],
        context_mask: Optional[torch.Tensor] = None,
    ) -> None:
        self._context = context
        self._context_mask = context_mask

    def clear_context(self) -> None:
        self._context = None
        self._context_mask = None

    def enable(self, flag: bool = True) -> None:
        self._enabled = bool(flag)

    def extra_repr(self) -> str:
        return f"n_layers={len(self.cross_layers)} hooks={len(self._hooks)}"


class EncoderMemoryProjector(nn.Module):
    """Project prim-encoder sequence (hidden) → AR dim for cross-attn K/V."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


__all__ = [
    "LayerCrossAttn",
    "DecoderCrossAttnInjector",
    "EncoderMemoryProjector",
]
