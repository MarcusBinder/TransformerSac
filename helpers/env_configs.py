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


# "current + exactly-2-steps-ago" for a per-turbine sensor: a length-3 deque
# [t-2, t-1, t] with window_length=1 reduces to [value_now, value_t-2].
# Mirrors NOW_AND_T2 from Example 7 (power tracking RL setup).
_NOW_AND_T2 = dict(
    current=False,
    rolling_mean=True,
    history_N=2,
    history_length=3,
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
    # [now, t-2]; wd/derate sensors are off. Fixed inflow (ws=10, wd=270,
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
        # Per-turbine sensors observed as [now, t-2].
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
}


def make_env_config(name: str = "default") -> Dict[str, Any]:
    """Build an env config by name. Applies overrides on top of the base config."""
    if name not in ENV_CONFIGS:
        available = ", ".join(sorted(ENV_CONFIGS.keys()))
        raise ValueError(f"Unknown env config '{name}'. Available: {available}")

    config = deepcopy(_base_config())
    return _deep_update(config, deepcopy(ENV_CONFIGS[name]))
