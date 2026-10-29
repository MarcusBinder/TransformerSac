"""``--env_overrides``: a JSON object deep-merged into the env config preset.

The trainer builds its WindFarmEnv config from a named preset
(``helpers/env_configs.py``) plus a handful of dedicated flags. Keys without a
flag (``derate_max``, ``derate_penalty``, ``track_def`` entries, ...) can be
changed per run with ``--env_overrides '{"derate_max": 1.0}'``; nested objects
merge key by key (``_deep_update``), so siblings in the preset are kept. The
flag lands in the checkpoint args like every other one, and
``eval_tracking_figs.build_track3_env`` applies it when rebuilding the env.
"""
from __future__ import annotations

import json
from typing import Optional

from helpers.env_configs import _deep_update


def parse_env_overrides(env_overrides: Optional[str]) -> dict:
    """JSON object -> dict; ``None``/empty -> {}. Raises ValueError on bad input."""
    if env_overrides is None or not str(env_overrides).strip():
        return {}
    try:
        obj = json.loads(env_overrides)
    except (TypeError, ValueError) as e:
        raise ValueError(f"env_overrides must be a JSON object, got {env_overrides!r}") from e
    if not isinstance(obj, dict):
        raise ValueError(f"env_overrides must be a JSON object, got {env_overrides!r}")
    return obj


def apply_env_overrides(config: dict, env_overrides: Optional[str]) -> None:
    """Deep-merge the overrides into ``config`` in place."""
    overrides = parse_env_overrides(env_overrides)
    if overrides:
        _deep_update(config, overrides)
