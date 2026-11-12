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
    """The turbine model a checkpoint was trained on, from its saved args.

    A missing or ``None`` ``derate_reference`` (runs trained without the flag,
    including every run before it existed) resolves to the preset's key
    (``available`` when the preset has none), so the spec always names the
    meaning the policy's derate action was trained with.
    """
    ref = args.get("derate_reference")
    if ref is None:
        from helpers.env_configs import make_env_config  # lazy: heavy module

        ref = make_env_config(args.get("config", "default")).get("derate_reference", "available")
    return {
        "derate_reference": str(ref).lower(),
        "turbine_params": parse_turbine_params(args.get("turbine_params", "{}")),
    }


def check_turbine_model(spec: dict[str, Any], facts) -> None:
    """Refuse an eval env whose derate meaning or filter constants differ from ``spec``.

    ``facts`` is the runner's static env description (``EnvFacts``); fields it
    does not carry, or the spec lacks (``derate_min`` / ``derate_max``), are not
    compared. Raises ``ValueError`` on a mismatch.
    """
    eval_ref = getattr(facts, "derate_reference", None)
    if eval_ref is not None and str(eval_ref).lower() != spec["derate_reference"]:
        raise ValueError(
            f"derate_reference differs from training: {eval_ref!r} vs "
            f"{spec['derate_reference']!r} (the derate action means something else; "
            "set derate_reference in the eval config)"
        )
    params = spec.get("turbine_params") or {}
    for key in ("tau_power", "tau_ct"):
        have = getattr(facts, key, None)
        if have is None:
            continue
        want = float(params.get(key, 0.0))
        if float(have) != want:
            raise ValueError(
                f"{key} differs from training: {have} vs {want} s "
                "(set turbine_params in the eval config)"
            )
    # What the derate action does (WindGym's share inner loop or the command
    # itself) and the yaw deadband: both change what the policy learned.
    if "derate_method" in spec:
        have = getattr(facts, "derate_method", None)
        if have is not None and str(have).lower() != str(spec["derate_method"]).lower():
            raise ValueError(
                f"derate_method differs from training: {have!r} vs {spec['derate_method']!r} "
                "(set it in the eval config's env_overrides)"
            )
    if "yaw_deadband" in spec:
        have = getattr(facts, "yaw_deadband", None)
        if have is not None and float(have) != float(spec["yaw_deadband"]):
            raise ValueError(
                f"yaw_deadband differs from training: {have} vs {spec['yaw_deadband']} deg "
                "(set it in the eval config's env_overrides)"
            )
    # The share inner loop's action meaning (raw shares or a tilt of proportional
    # shares) and reference (observed or scored), and the yaw hold.
    for key in ("share_mapping", "share_reference"):
        have = getattr(facts, key, None)
        if key in spec and have is not None and str(have).lower() != str(spec[key]).lower():
            raise ValueError(
                f"{key} differs from training: {have!r} vs {spec[key]!r} "
                "(set it in the eval config's env_overrides)"
            )
    if "yaw_hold_s" in spec:
        have = getattr(facts, "yaw_hold_s", None)
        if have is not None and float(have) != float(spec["yaw_hold_s"]):
            raise ValueError(
                f"yaw_hold_s differs from training: {have} vs {spec['yaw_hold_s']} s "
                "(set it in the eval config's env_overrides)"
            )
    # The agent step (seconds per env step) and the derate slew per sim step:
    # a policy evaluated at another cadence or slew runs silently but wrongly.
    if "dt_env" in spec:
        have = getattr(facts, "dt_env", None)
        if have is not None and float(have) != float(spec["dt_env"]):
            raise ValueError(
                f"dt_env differs from training: {have} vs {spec['dt_env']} s "
                "(set dt_env in the eval config)"
            )
    if "derate_step_sim" in spec and hasattr(facts, "derate_step_sim"):
        have, want = facts.derate_step_sim, spec["derate_step_sim"]
        if (have is None) != (want is None) or (
            have is not None and float(have) != float(want)
        ):
            raise ValueError(
                f"derate_step_sim differs from training: {have} vs {want} "
                "(set it in the eval config's env_overrides)"
            )
    # The derate action range: the policy's [-1, 1] maps onto it.
    for key in ("derate_min", "derate_max"):
        have = getattr(facts, key, None)
        if have is None or key not in spec:
            continue
        if float(have) != float(spec[key]):
            raise ValueError(
                f"{key} differs from training: {have} vs {spec[key]} "
                "(set it in the eval config: a policy is evaluated with its own action range)"
            )
