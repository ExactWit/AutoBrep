"""Tests for P1-B decoder cross-attn injector."""

from __future__ import annotations

import torch
import torch.nn as nn

from autobrep.models.decoder_cross_attn import (
    DecoderCrossAttnInjector,
    EncoderMemoryProjector,
    LayerCrossAttn,
)


class _FakeAttn(nn.Module):
    def forward(self, x, **kwargs):
        return x * 1.0, {"ok": True}


class _FakeAttnLayers(nn.Module):
    def __init__(self, dim=32, n=2):
        super().__init__()
        self.layer_types = tuple(["a", "f"] * n)
        self.depth = n
        blocks = []
        for t in self.layer_types:
            if t == "a":
                blocks.append((nn.Identity(), _FakeAttn(), nn.Identity()))
            else:
                blocks.append((nn.Identity(), nn.Linear(dim, dim), nn.Identity()))
        self.layers = nn.ModuleList(
            [nn.ModuleList([a, b, c]) for a, b, c in blocks]
        )


def test_layer_cross_attn_zero_init_identity():
    m = LayerCrossAttn(32, n_heads=4)
    x = torch.randn(2, 5, 32)
    ctx = torch.randn(2, 7, 32)
    mask = torch.ones(2, 7, dtype=torch.bool)
    y = m(x, ctx, context_mask=mask)
    assert torch.allclose(y, x, atol=1e-5)


def test_injector_hooks():
    dim = 32
    fake = _FakeAttnLayers(dim=dim, n=2)
    inj = DecoderCrossAttnInjector(fake, dim=dim, n_heads=4)
    assert len(inj.cross_layers) == 2
    assert len(inj._hooks) == 2
    ctx = torch.randn(1, 4, dim)
    mask = torch.ones(1, 4, dtype=torch.bool)
    inj.set_context(ctx, mask)
    # run a hooked block
    out, _ = fake.layers[0][1](torch.randn(1, 3, dim))
    assert out.shape == (1, 3, dim)
    # grads flow into cross layers
    loss = out.sum()
    loss.backward()
    assert any(p.grad is not None for p in inj.cross_layers.parameters())


def test_mem_proj():
    p = EncoderMemoryProjector(16, 32)
    x = torch.randn(2, 8, 16)
    y = p(x)
    assert y.shape == (2, 8, 32)
