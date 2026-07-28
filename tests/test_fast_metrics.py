"""Unit tests for Level-1 fast metrics (no GPU / OCC)."""

from __future__ import annotations

import torch

from autobrep.data.token_mapping import MMTokenIndex
from autobrep.metrics.fast_metrics import (
    compute_fast_metrics_from_logits,
    light_topo_compliance,
)


def test_light_topo_balanced():
    seq = torch.tensor(
        [
            MMTokenIndex.BOS,
            MMTokenIndex.BOM,
            MMTokenIndex.GEN_MID,
            MMTokenIndex.EOM,
            MMTokenIndex.BOC,
            MMTokenIndex.BOL,
            MMTokenIndex.BOF,
            MMTokenIndex.EOF,
            MMTokenIndex.EOL,
            MMTokenIndex.EOC,
            MMTokenIndex.EOS,
        ],
        dtype=torch.long,
    )
    assert light_topo_compliance(seq) == 1.0


def test_light_topo_unbalanced():
    seq = torch.tensor([MMTokenIndex.BOF, MMTokenIndex.BOF, MMTokenIndex.EOF], dtype=torch.long)
    assert light_topo_compliance(seq) == 0.0


def test_fast_metrics_from_logits():
    b, t, v = 2, 8, 64
    logits = torch.zeros(b, t, v)
    targets = torch.randint(0, v, (b, t))
    # Make argmax match targets on first row
    for i in range(t):
        logits[0, i, int(targets[0, i])] = 10.0
    cond = torch.zeros(b, t, dtype=torch.bool)
    cond[:, :1] = True
    m = compute_fast_metrics_from_logits(logits, targets, cond_mask=cond)
    assert m.n_tokens > 0
    assert m.ppl == m.ppl  # not nan
    assert 0.0 <= m.token_acc <= 1.0
