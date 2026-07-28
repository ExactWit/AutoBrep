"""Lightning callback: dump Level-1 fast metrics JSON each validation epoch."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from lightning import Callback, LightningModule, Trainer


class FastValMetricsCallback(Callback):
    """
    After each validation epoch, write aggregated ``val/fast/*`` metrics to
    ``{exp_dir}/metrics/fast_val_epoch{NNN}.json``.

    Relies on AutoBrepViewModel logging ``val/fast/*`` during ``validation_step``.
    """

    def __init__(self, metrics_dir: str | Path, enabled: bool = True) -> None:
        super().__init__()
        self.metrics_dir = Path(metrics_dir)
        self.enabled = enabled

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if not self.enabled or trainer.sanity_checking:
            return
        metrics = trainer.callback_metrics
        payload: dict[str, Any] = {
            "epoch": int(trainer.current_epoch),
            "global_step": int(trainer.global_step),
        }
        for key, val in metrics.items():
            sk = str(key)
            if not sk.startswith("val/fast"):
                continue
            try:
                payload[sk] = float(val.item() if hasattr(val, "item") else val)
            except Exception:
                payload[sk] = str(val)
        if len(payload) <= 2:
            return
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        out = self.metrics_dir / f"fast_val_epoch{int(trainer.current_epoch) + 1:03d}.json"
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"[fast_metrics] wrote {out}", flush=True)
