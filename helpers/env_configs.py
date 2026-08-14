from copy import deepcopy
from typing import Dict, Any


def _base_config() -> Dict[str, Any]:
    """Base environment configuration for transformer-based control."""
    return {
        "yaw_init": "Random",
        "BaseController": "Local",
        "ActionMethod": "yaw",
        "Track_power": False,
        "farm": {"yaw_min": -30, "yaw_max": 30},
        "wind": {
            "ws_min": 10, "ws_max": 10,
            "TI_min": 0.07, "TI_max": 0.07,
            "wd_min": 260, "wd_max": 280,
        },
        "act_pen": {"action_penalty": 0.0, "action_penalty_type": "Change"},
        "power_def": {"Power_reward": "Baseline", "Power_avg": 1, "Power_scaling": 1.0},
        "mes_level": {
            "turb_ws": True,
            "turb_wd": True,
            "turb_TI": False,
            "turb_power": True,
            "farm_ws": False,
            "farm_wd": False,
            "farm_TI": False,
            "farm_power": False,
        },
        "ws_mes": {
            "ws_current": False,
            "ws_rolling_mean": True,
            "ws_history_N": 15,
            "ws_history_length": 15,
            "ws_window_length": 1,
        },
        "wd_mes": {
            "wd_current": False,
            "wd_rolling_mean": True,
            "wd_history_N": 15,
            "wd_history_length": 15,
            "wd_window_length": 1,
        },
        "yaw_mes": {
            "yaw_current": False,
            "yaw_rolling_mean": True,
            "yaw_history_N": 15,
            "yaw_history_length": 15,
            "yaw_window_length": 1,
        },
        "power_mes": {
            "power_current": False,
            "power_rolling_mean": True,
            "power_history_N": 15,
            "power_history_length": 15,
            "power_window_length": 1,
        },
    }


def _deep_update(base: dict, overrides: dict) -> dict:
    """Recursively update base dict with overrides."""
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


# 3 evenly-spaced samples over a ~10-step window for a per-turbine sensor: a
# length-10 deque [t-9, ..., t] with history_N=3, window_length=1 reduces to
# [t, t-5, t-9] (widened from the earlier 2-samples-over-3-steps [t, t-2] to
# carry more temporal history on the dynamic dynamiks/DWM backend).
# Mirrors (and widens) NOW_AND_T2 from Example 7 (power tracking RL setup).
_NOW_AND_T2 = dict(
    current=False,
    rolling_mean=True,
    history_N=3,
    history_length=10,
    window_length=1,
)


def _now_and_t2(prefix: str) -> Dict[str, Any]:
    """Per-sensor NOW_AND_T2 measurement block, e.g. ws_current, ws_rolling_mean."""
    return {f"{prefix}_{k}": v for k, v in _NOW_AND_T2.items()}


# Registry: name -> overrides from base
ENV_CONFIGS: Dict[str, Dict[str, Any]] = {
    "default": {},
    "big": {
        "BaseController": "PyWake",
        "wind": {
            "ws_min": 9, "ws_max": 9,
            "wd_min": 225, "wd_max": 315
        },
    },
    # Easy to add more:
    "hard": {
        "wind": {
            "wd_min": 225, "wd_max": 315,
            "ws_min": 10, "ws_max": 14,
        },
    },
    "hard_2": {
        "power_def": {"Power_reward": "Wake_recovery", "Power_avg": 5, "Power_scaling": 1.0},
        "wind": {
            "wd_min": 225, "wd_max": 315,
            "ws_min": 10, "ws_max": 14,
        },
    },

    # hard_2 with wind direction removed from the per-turbine observation.
    # Probes whether wind-frame profile ROTATION alone is sufficient for wd
    # perception: rotation reads env.wd directly (a privileged, noiseless,
    # instantaneous channel), so it is unaffected by dropping the lagged
    # rolling-mean wd from the obs. Removing turb_wd costs exactly
    # history_length features per turbine (wd_rolling_mean with wd_history_N =
    # history_length), and EnhancedPerTurbineWrapper degrades to a pass-through
    # because _detect_wd_indices returns None when turb_wd is False.
    "hard_2_nowd": {
        "power_def": {"Power_reward": "Wake_recovery", "Power_avg": 5, "Power_scaling": 1.0},
        "wind": {
            "wd_min": 225, "wd_max": 315,
            "ws_min": 10, "ws_max": 14,
        },
        "mes_level": {"turb_wd": False},
    },

    "basic": {
        "yaw_init": "Random",
        "BaseController": "Local",
        "ActionMethod": "yaw",
        "Track_power": False,
        "farm": {"yaw_min": -30, "yaw_max": 30},
        "wind": {
            "ws_min": 10, "ws_max": 14,
            "TI_min": 0.07, "TI_max": 0.07,
            "wd_min": 225, "wd_max": 315,
        },
        "act_pen": {"action_penalty": 0.0, "action_penalty_type": "Change"},
        "power_def": {"Power_reward": "Power_avg", "Power_avg": 1, "Power_scaling": 1.0},
        "mes_level": {
            "turb_ws": True,
            "turb_wd": True,
            "turb_TI": False,
            "turb_power": True,
            "farm_ws": False,
            "farm_wd": False,
            "farm_TI": False,
            "farm_power": False,
        },
        "ws_mes": {
            "ws_current": True,
            "ws_rolling_mean": False,
            "ws_history_N": 1,
            "ws_history_length": 1,
            "ws_window_length": 1,
        },
        "wd_mes": {
            "wd_current": True,
            "wd_rolling_mean": False,
            "wd_history_N": 15,
            "wd_history_length": 15,
            "wd_window_length": 1,
        },
        "yaw_mes": {
            "yaw_current": True,
            "yaw_rolling_mean": False,
            "yaw_history_N": 1,
            "yaw_history_length": 1,
            "yaw_window_length": 1,
        },
        "power_mes": {
            "power_current": True,
            "power_rolling_mean": False,
            "power_history_N": 1,
            "power_history_length": 1,
            "power_window_length": 1,
        },
    },

    "wide": {
        "wind": {
            "wd_min": 250, "wd_max": 290,
            "ws_min": 10, "ws_max": 10,
            "TI_min": 0.07, "TI_max": 0.07,
        },
    },

    "20deg_wd": {
        "wind": {
            "wd_min": 250, "wd_max": 290,
        },
    },

    # Derate-only farm power tracking. Mirrors Example 7's make_config(): the
    # agent commands an absolute per-turbine derate in [0, 0.8] to track a
    # farm-power reference (supplied via WindFarmEnv(power_ref_function=...)).
    # The tracking reward requires the power-maximization reward off
    # (power_def.Power_reward == "None"). Per-turbine sensors are observed as
    # [t, t-5, t-9]; wd/derate sensors are off. Fixed inflow (ws=10, wd=270,
    # TI=0.06) so a one-off greedy probe is valid for the whole run.
    # yaw_init is forced off (base defaults it to "Random", which would
    # randomize the fixed yaw of a derate-only farm).
    "power_tracking": {
        "yaw_init": None,
        "ActionMethod": "wind",
        "Track_power": True,
        "yaw_action": False,        # derate-only agent
        "derate_action": True,
        "derate_min": 0.0,
        "derate_max": 0.8,
        "derate_method": "absolute",
        "track_def": {
            "Track_reward": "abs",  # r = -|P_farm - P_ref| / (rated farm power)
            "track_obs_setpoint": True,
            "track_obs_error": True,
            "track_obs_preview": 0,
        },
        "farm": {"yaw_min": -30, "yaw_max": 30},
        "wind": {
            "ws_min": 10.0, "ws_max": 10.0,
            "TI_min": 0.06, "TI_max": 0.06,
            "wd_min": 270.0, "wd_max": 270.0,
        },
        "act_pen": {"action_penalty": 0.0, "action_penalty_type": "change"},
        # Track_power requires the power-maximization reward to be off.
        "power_def": {"Power_reward": "None", "Power_avg": 5, "Power_scaling": 1.0},
        "mes_level": {
            "turb_ws": True,
            "turb_wd": False,
            "turb_TI": False,
            "turb_power": True,
            "farm_ws": False,
            "farm_wd": False,
            "farm_TI": False,
            "farm_power": False,
        },
        # Per-turbine sensors observed as [t, t-5, t-9] (3 samples / ~10 steps).
        "ws_mes": _now_and_t2("ws"),
        "power_mes": _now_and_t2("power"),
        "yaw_mes": _now_and_t2("yaw"),
        # wd sensor required by the schema but switched off.
        "wd_mes": {
            "wd_current": False,
            "wd_rolling_mean": False,
            "wd_history_N": 0,
            "wd_history_length": 3,
            "wd_window_length": 1,
        },
        # Derate sensor explicitly off. To feed the agent its own recent derate,
        # flip these to the NOW_AND_T2 pattern (_now_and_t2("derate")).
        "derate_mes": {"derate_current": False, "derate_rolling_mean": False},
    },

    # Yaw+derate power MAXIMIZATION (not tracking) for the DEL-penalty A/B
    # experiment on square_2x2. Inherits the base Baseline reward
    # (P_agent/P_baseline - 1), which forces Baseline_comp=True, so the greedy
    # baseline farm exists in every run -- both the reward denominator and the
    # DELRewardWrapper's DEL reference come for free and cost the same with the
    # penalty on or off. derate_max=0.2 caps the derate at 20% (pset >= 0.8),
    # the DEL surrogate's validity range, so derating alone can never push the
    # surrogate out of distribution. BaseController="Global" pins the baseline
    # farm at zero yaw offset (deterministic greedy reference); the base
    # default "Local" would chase wake-perturbed local wind directions and add
    # noise to both the reward and the DEL ratio. Fixed inflow (ws=10, wd=270,
    # TI=0.06) matches the power_tracking presets.
    # LES-3x3 campaign (270 -> 235 scenario on les_3x3, no privileged wd).
    # hard_2's reward (Wake_recovery, Power_avg 5) + the LESRL "good recipe"
    # physical envelope: yaw band +-45 (hard_2's +-30 is tight against a 35 deg
    # swing — nacelles are world-fixed under frame rotation, so frame-relative
    # yaw drifts by the full 35 deg during the transition), ws 8-11 (brackets
    # the scenario's nominal 9-10; DTU10MW rates at 11.4 so headroom exists
    # everywhere), TI 0.038 fixed (LESRL value). wd domain stays [225, 315] so
    # dr_* train schedules compose unchanged. The preset carries the physical
    # envelope (no CLI flags exist for yaw band / TI); the campaign's CONTRAST
    # levers stay CLI flags (--action_penalty, --max_turb_move,
    # --train_wd_function, ws overrides).
    "les_recipe": {
        "power_def": {"Power_reward": "Wake_recovery", "Power_avg": 5, "Power_scaling": 1.0},
        "farm": {"yaw_min": -45, "yaw_max": 45},
        "wind": {
            "wd_min": 225, "wd_max": 315,
            "ws_min": 8, "ws_max": 11,
            "TI_min": 0.038, "TI_max": 0.038,
        },
    },

    "power_max_derate": {
        "BaseController": "Global",
        "yaw_action": True,
        "derate_action": True,
        "derate_min": 0.0,
        "derate_max": 0.2,          # pset in [0.8, 1.0] = DEL surrogate validity
        "derate_method": "absolute",
        "wind": {"ws_min": 10.0, "ws_max": 10.0, "TI_min": 0.06, "TI_max": 0.06,
                 "wd_min": 270.0, "wd_max": 270.0},
        "derate_mes": {"derate_current": False, "derate_rolling_mean": False},
    },
}


# power_max_derate with the derate SENSOR enabled (same 15-sample pattern as
# the base ws/yaw/power sensors; history_N/length are overridden to
# --history_length by the trainer like every other per-turbine sensor).
# Needed once derate_step_sim rate-limits the derate: the realized derate
# then lags the commanded setpoint and becomes a genuine (otherwise hidden)
# state, exactly like the slewed yaw whose sensor has always been on.
# Cloned instead of editing power_max_derate in place so the finished
# delmax_2x2 runs remain reproducible from their pinned preset.
ENV_CONFIGS["power_max_derate_dobs"] = _deep_update(
    deepcopy(ENV_CONFIGS["power_max_derate"]),
    {"derate_mes": {
        "derate_current": False,
        "derate_rolling_mean": True,
        "derate_history_N": 15,
        "derate_history_length": 15,
        "derate_window_length": 1,
    }},
)


# Yaw+derate farm power tracking: identical to "power_tracking" but with yaw
# added as a second action channel (yaw_action=True -> act_var=2, block action
# [yaw..., derate...]). This is the config the >100% BOOST_SCHEDULE requires:
# the agent must steer wakes (yaw) to boost farm power above the no-steering
# greedy baseline, since derating alone can only reduce it. Cloned from
# "power_tracking" so it tracks any future change to that preset; only
# yaw_action is overridden (yaw bounds +-30, yaw_mes and fixed inflow all
# inherited). The yaw command style is set at the trainer via --action_type
# (config["ActionMethod"] = args.action_type); the launcher uses --action_type
# yaw (delta yaw). Derate stays an absolute setpoint (derate_method="absolute").
ENV_CONFIGS["power_tracking_yaw"] = _deep_update(
    deepcopy(ENV_CONFIGS["power_tracking"]),
    {"yaw_action": True},
)


def make_env_config(name: str = "default") -> Dict[str, Any]:
    """Build an env config by name. Applies overrides on top of the base config."""
    if name not in ENV_CONFIGS:
        available = ", ".join(sorted(ENV_CONFIGS.keys()))
        raise ValueError(f"Unknown env config '{name}'. Available: {available}")

    config = deepcopy(_base_config())
    return _deep_update(config, deepcopy(ENV_CONFIGS[name]))


# CLI flag -> (config section, key). Every one of these defaults to None on Args,
# meaning "don't override", so a script that passes none of them gets exactly the
# preset's values and pre-change_wd_3 behaviour is bit-identical.
_OVERRIDE_MAP = (
    ("reward_tau",    "power_def", "tau"),
    ("power_reward",  "power_def", "Power_reward"),
    ("power_avg",     "power_def", "Power_avg"),
    ("power_scaling", "power_def", "Power_scaling"),
    ("train_ws_min",  "wind",      "ws_min"),
    ("train_ws_max",  "wind",      "ws_max"),
    ("action_penalty",      "act_pen", "action_penalty"),
    ("action_penalty_type", "act_pen", "action_penalty_type"),
)


def apply_config_overrides(config: Dict[str, Any], args) -> Dict[str, Any]:
    """Apply the change_wd_3 reward / training-wind CLI overrides in place.

    Used so the 15-arm sweep can request arbitrary combinations of
    (tau x Power_avg x Power_scaling x Power_reward x training ws range) without a
    dozen near-duplicate ENV_CONFIGS presets.

    MUST be called before the eval configs are deep-copied off ``config``: the
    reward overrides are meant to reach the eval envs too (the evaluator's
    baseline-power bookkeeping shares this config), while the ws override is
    training-only and is undone for eval by re-pinning ws_min/ws_max per eval spec.

    Returns the same dict, mutated, and prints every applied override so the SLURM
    log records the arm's actual reward definition.
    """
    applied = []
    for attr, section, key in _OVERRIDE_MAP:
        value = getattr(args, attr, None)
        if value is None:
            continue
        config.setdefault(section, {})
        before = config[section].get(key)
        config[section][key] = value
        applied.append(f"{section}.{key}: {before} -> {value}")
    if applied:
        print("Config overrides: " + "; ".join(applied))
    return config


def make_eval_wind_config(config: Dict[str, Any], wd0: float, ws: float) -> Dict[str, Any]:
    """Deep-copy ``config`` and pin its wind to one eval condition.

    ``wd0`` is ``wd_function(0)``: the env's burn-in holds the per-reset base_wd,
    so pinning wd there is what prevents a wind-direction jump at t=0 when the
    schedule takes over (pattern: make_flow_gif.py).

    ws is pinned UNCONDITIONALLY, which is the guard that keeps a --train_ws_min /
    --train_ws_max override -- applied to the shared config upstream -- from
    leaking into the eval condition. Everything else (notably power_def) is
    inherited, so the eval envs share the arm's reward definition.
    """
    eval_config = deepcopy(config)
    eval_config["wind"]["wd_min"] = eval_config["wind"]["wd_max"] = float(wd0)
    eval_config["wind"]["ws_min"] = eval_config["wind"]["ws_max"] = float(ws)
    return eval_config
