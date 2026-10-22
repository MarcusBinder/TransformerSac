"""Append-only JSON-lines event log for the trainer (``--metrics_jsonl``).

The LESRL optuna sweep runs the trainer as a subprocess and needs its eval
results WITHOUT parsing TensorBoard event files: the worker tails this file
(byte offset, tolerant of a torn last line) and feeds the smoothed score to
the pruner. One JSON object per line, ``flush`` + ``fsync`` after every write
so a reader on the same node (or a SIGKILLed trainer) never sees a half line
except the one being written.

Events written by transformer_sac_windfarm.py:
  {"event": "started",  "time", "pid", "obs_dim", "n_turbines", "n_params_actor", ...}
  {"event": "eval",     "time", "step", **eval_dict}        (every in-training eval)
  {"event": "finished", "time", "step", "stop_requested"}   (after the final save)
  {"event": "baseline", "time", "policy", **eval_dict}      (--baseline_eval_policy)

``MetricsJsonl(None)`` is a no-op so call sites need no guards.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Optional

import numpy as np


def _to_jsonable(v: Any) -> Any:
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (bool, int, float, str)) or v is None:
        return v
    if isinstance(v, dict):
        return {str(k): _to_jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_to_jsonable(x) for x in v]
    return str(v)


class MetricsJsonl:
    """Append one JSON line per event to ``path`` (None = disabled)."""

    def __init__(self, path: Optional[str]):
        self.path = path
        if path:
            os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)

    @property
    def enabled(self) -> bool:
        return bool(self.path)

    def write(self, event: str, **fields: Any) -> None:
        if not self.path:
            return
        rec = {"event": str(event), "time": time.time()}
        rec.update({str(k): _to_jsonable(v) for k, v in fields.items()})
        line = json.dumps(rec, separators=(",", ":")) + "\n"
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())
