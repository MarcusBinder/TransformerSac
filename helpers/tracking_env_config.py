"""The tracking trainer's env config: preset plus every override layer, in order.

Shared by ``transformer_sac_windfarm_tracking.py`` and
``eval_tracking_figs.build_track3_env`` so a checkpoint's env is rebuilt with the
same layers it was trained with:

1. the named preset (``--config``), with ``ActionMethod`` from ``--action_type``;
2. the turbine model (``--derate_reference``, ``--turbine_params``);
3. the reward / wind / yaw-band flags of ``_OVERRIDE_MAP`` (``--power_avg``,
   ``--power_scaling``, ``--action_penalty``, ...; None = keep the preset);
4. ``--env_overrides``, the most specific layer.

Until this helper existed the tracking trainer skipped layer 3, so those flags
were parsed, saved in the checkpoint args and silently dropped.
"""
from __future__ import annotations

from helpers.env_configs import _arg, apply_config_overrides, make_env_config
from helpers.env_overrides import apply_env_overrides
from helpers.turbine_model import apply_turbine_model


def build_tracking_env_config(args) -> tuple[dict, dict]:
    """Return ``(config, turbine_model_kwargs)`` for an ``Args`` or a checkpoint-args dict.

    ``turbine_model_kwargs`` holds the WindFarmEnv constructor kwargs of the
    turbine model (``dwm_params``), to be merged into the env kwargs.
    """
    config = make_env_config(_arg(args, "config"))
    config["ActionMethod"] = _arg(args, "action_type")
    turbine_model_kwargs: dict = {}
    apply_turbine_model(
        config, turbine_model_kwargs,
        derate_reference=_arg(args, "derate_reference"),
        turbine_params=_arg(args, "turbine_params"),
    )
    apply_config_overrides(config, args)
    apply_env_overrides(config, _arg(args, "env_overrides"))
    return config, turbine_model_kwargs
