"""Unit tests for P1-A PrimTransformerEncoder soft prefix path."""

from __future__ import annotations

import torch

from autobrep.models.prim_transformer_encoder import (
    PrimTransformerEncoder,
    SoftPrefixCompressor,
    quantize_geom,
)
from autobrep.models.view_condition_encoder import ViewConditionEncoder


def test_quantize_geom_range():
    x = torch.tensor([[[-1.0, 0.0, 1.0]]])
    q = quantize_geom(x, n_bins=1024)
    assert int(q[0, 0, 0]) == 0
    assert int(q[0, 0, 2]) == 1023


def test_prim_encoder_forward():
    enc = PrimTransformerEncoder(d_model=64, n_layers=2, n_heads=4, max_seq=32, out_dim=32)
    b, v, n = 2, 3, 16
    types = torch.randint(0, 7, (b, v, n))
    lts = torch.randint(0, 4, (b, v, n))
    geom = torch.randn(b, v, n, 12)
    mask = torch.zeros(b, v, n, dtype=torch.bool)
    mask[:, :, :8] = True
    roles = torch.zeros(b, v, n, dtype=torch.long)
    tokens, m = enc(types, lts, geom, mask, prim_group_roles=roles)
    assert tokens.shape[0] == b
    assert tokens.shape[-1] == 32
    assert m.shape[0] == b


def test_view_encoder_prim_seq_mode():
    enc = ViewConditionEncoder(
        dim=128,
        hidden=64,
        num_latents=8,
        num_heads=4,
        use_prim_seq_encoder=True,
        prim_d_model=64,
        prim_n_layers=2,
        prim_max_seq=48,
        pretrained_backbone=False,
    )
    b = 2
    images = torch.rand(b, 3, 3, 32, 32)
    types = torch.randint(0, 7, (b, 3, 16))
    lts = torch.randint(0, 4, (b, 3, 16))
    geom = torch.randn(b, 3, 16, 12)
    mask = torch.ones(b, 3, 16, dtype=torch.bool)
    roles = torch.zeros(b, 3, 16, dtype=torch.long)
    enc.eval()
    out = enc(images, types, lts, geom, mask, prim_group_roles=roles)
    assert out.shape == (b, 8 + 2, 128)
    enc.train()
    out = enc(images, types, lts, geom, mask, prim_group_roles=roles)
    loss = out.sum()
    loss.backward()
    # Encoder params should get grads (img path always; prim when not dropped)
    assert any(
        p.grad is not None
        for p in list(enc.prim_encoder.parameters()) + list(enc.img_proj.parameters())
    )
