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
