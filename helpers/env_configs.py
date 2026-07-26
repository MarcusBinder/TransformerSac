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
}


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
