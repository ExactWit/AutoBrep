"""Training / eval metrics (fast token-level and optional OCC)."""

from autobrep.metrics.fast_metrics import (
    FastMetricResult,
    bucket_token_mask,
    compute_fast_metrics_from_logits,
    light_topo_compliance,
)

__all__ = [
    "FastMetricResult",
    "bucket_token_mask",
    "compute_fast_metrics_from_logits",
    "light_topo_compliance",
]
