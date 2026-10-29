"""Turbine-model flags -> WindFarmEnv config + constructor kwargs.

``--derate_reference`` (None = keep the preset's key; "available" | "rated" |
"setpoint") overrides the env config's derate semantics, and
``--turbine_params`` (a JSON object of WindGym ``TURBINE_PARAM_SPECS`` keys,
e.g. ``'{"tau_power": 6, "tau_ct": 10}'``) becomes the ``dwm_params``
constructor kwarg (a dict of floats: picklable for ``AsyncVectorEnv``).

Shared by the tracking trainer (train + greedy probe env), ``eval_tracking_figs``
(the rebuilt training env) and ``eval_powertrack`` (the checkpoint's spec), so
every env built for a checkpoint carries the turbine model it was trained on.
"""
from __future__ import annotations

import json
from typing import Any, Optional

VALID_REFERENCES = ("available", "rated", "setpoint")


def parse_turbine_params(turbine_params: Optional[str]) -> dict[str, float]:
    """JSON object -> {key: float}; ``None``/empty -> {}. Raises ValueError on bad input."""
    if turbine_params is None or not str(turbine_params).strip():
        return {}
    try:
        obj = json.loads(turbine_params)
    except (TypeError, ValueError) as e:
        raise ValueError(f"turbine_params must be a JSON object, got {turbine_params!r}") from e
    if not isinstance(obj, dict):
        raise ValueError(f"turbine_params must be a JSON object, got {turbine_params!r}")
    out: dict[str, float] = {}
    for k, v in obj.items():
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError(f"turbine_params[{k!r}] must be a number, got {v!r}")
        out[str(k)] = float(v)
    return out


def apply_turbine_model(
    config: dict, env_kwargs: dict, *, derate_reference: Optional[str], turbine_params: Optional[str]
) -> None:
    """Apply the overrides in place: ``config["derate_reference"]`` and
    ``env_kwargs["dwm_params"]`` (merged into an existing mapping)."""
    if derate_reference is not None:
        ref = str(derate_reference).lower()
        if ref not in VALID_REFERENCES:
            raise ValueError(f"derate_reference must be one of {VALID_REFERENCES}, got {derate_reference!r}")
        config["derate_reference"] = ref
    params = parse_turbine_params(turbine_params)
    if params:
        env_kwargs["dwm_params"] = {**(env_kwargs.get("dwm_params") or {}), **params}


def turbine_model_spec(args: dict[str, Any]) -> dict[str, Any]:
    """The two fields as saved in a checkpoint's args (back-filled for old runs)."""
    return {
        "derate_reference": args.get("derate_reference"),
        "turbine_params": parse_turbine_params(args.get("turbine_params", "{}")),
    }
