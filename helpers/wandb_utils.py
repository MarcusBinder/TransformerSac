"""W&B run start for the trainer (``--wandb_soft_fail``, LESRL optuna_sweep).

Without soft-fail ``init_wandb`` is exactly the old ``wandb.init(**kw)``: any
error propagates. With soft-fail (the optuna worker passes it) W&B trouble can
never kill a trial: a failed online init (bounded by WANDB_INIT_TIMEOUT, set by
the worker) retries offline (the worker stages the offline run dir for a later
``wandb sync``), and if that fails too the run continues untracked
(``args.track = False``). No explicit ``wandb.Settings`` here: building one
re-validates every WANDB_* env var (wandb 0.21 rejected a WANDB_CONFIG_PATHS
string there, failing every online init).

``--wandb_init_json`` (``wandb_init_kwargs``) passes the worker's run id, tags
and extra config (optuna/*) as explicit ``wandb.init`` arguments; the WANDB_TAGS
/ WANDB_CONFIG_PATHS env vars were not applied by wandb 0.21.
"""
from __future__ import annotations

import json
from typing import Optional

# Everything reaches W&B through sync_tensorboard, so the TB tag is the W&B key
# and the TB step arrives as "global_step"; these groups chart against it.
STEP_METRIC_GROUPS = ("perf/*", "actions/*", "del/*", "losses/*",
                      "charts/*", "entropy/*", "timing/*", "eval/*", "eval_dr/*")


def wandb_init_kwargs(init_json: Optional[str], config: dict) -> dict:
    """``config`` (+ ``id`` / ``tags`` / extra config from ``init_json``) for wandb.init.
    The trainer's own config keys win a clash; None = just the config (the old call)."""
    if not init_json:
        return {"config": config}
    with open(init_json) as f:
        extra = json.load(f)
    kw = {"config": {**extra.get("config", {}), **config}}
    for k in ("id", "tags"):
        if extra.get(k) is not None:
            kw[k] = extra[k]
    return kw


def init_wandb(args, *, soft_fail: bool = False, **kw):
    """``wandb.init(**kw)``; returns the run, or None after a soft-fail give-up."""
    if not soft_fail:
        import wandb
        return wandb.init(**kw)
    try:
        import wandb
        return wandb.init(**kw)
    except Exception as e:  # noqa: BLE001 - any W&B failure must not end the trial
        print(f"[wandb] online init failed ({e!r}); retrying offline", flush=True)
    try:
        return wandb.init(mode="offline", **kw)
    except Exception as e:  # noqa: BLE001
        print(f"[wandb] offline init failed too ({e!r}); continuing without W&B", flush=True)
        args.track = False
        return None
