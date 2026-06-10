from copy import deepcopy
from typing import Dict, Any


def _base_config() -> Dict[str, Any]:
    """Base environment configuration for transformer-based control."""
    return {
        "yaw_init": "Random",
        "BaseController": "Local",
        "ActionMethod": "yaw",
        "Track_power": False,
        "farm": {"yaw_min": -45, "yaw_max": 45},
        "wind": {
            "wd_min": 270, "wd_max": 270,
            "ws_min": 8, "ws_max": 11,
            "TI_min": 0.038, "TI_max": 0.038,
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
    "LES": {
        "wind": {
            "wd_min": 270, "wd_max": 270,
            "ws_min": 8, "ws_max": 11,
            "TI_min": 0.038, "TI_max": 0.038,
        },
        "farm": {"yaw_min": -45, "yaw_max": 45},
    },
    "test": {
        "wind": {
            "wd_min": 270, "wd_max": 270,
            "ws_min": 10, "ws_max": 10,
            "TI_min": 0.038, "TI_max": 0.038,
        },
        "farm": {"yaw_min": -45, "yaw_max": 45},
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


# Sensor-ablation variants of the LES config. Yaw is always present (WindGym's
# FarmMes never gates yaw; mes_class.py only ANDs the turb_* flags into the
# ws/wd/power channels). Only the turb_* flags differ from "LES"; turb_TI and
# the farm_* flags keep their base values (all False) via the deep merge.
_LES_ABLATIONS = {
    # yaw + one sensor (build-up)
    "LES_yaw":      {"turb_ws": False, "turb_wd": False, "turb_power": False},
    "LES_yawpower": {"turb_ws": False, "turb_wd": False, "turb_power": True},
    "LES_yawws":    {"turb_ws": True,  "turb_wd": False, "turb_power": False},
    "LES_yawwd":    {"turb_ws": False, "turb_wd": True,  "turb_power": False},
    # full set minus one sensor (leave-one-out)
    "LES_nows":     {"turb_ws": False, "turb_wd": True,  "turb_power": True},
    "LES_nowd":     {"turb_ws": True,  "turb_wd": False, "turb_power": True},
    "LES_nopower":  {"turb_ws": True,  "turb_wd": True,  "turb_power": False},
}
for _name, _mes in _LES_ABLATIONS.items():
    _cfg = deepcopy(ENV_CONFIGS["LES"])
    _cfg["mes_level"] = dict(_mes)
    ENV_CONFIGS[_name] = _cfg


def make_env_config(name: str = "default") -> Dict[str, Any]:
    """Build an env config by name. Applies overrides on top of the base config."""
    if name not in ENV_CONFIGS:
        available = ", ".join(sorted(ENV_CONFIGS.keys()))
        raise ValueError(f"Unknown env config '{name}'. Available: {available}")

    config = deepcopy(_base_config())
    return _deep_update(config, deepcopy(ENV_CONFIGS[name]))


def obs_dim_per_turbine(config: Dict[str, Any], history_length: int) -> int:
    """Per-turbine observation size produced by WindGym's FarmMes for this config.

    Mirrors TurbMes.observed_variables(): the ws/wd/power channels are gated by
    the mes_level turb_* flags (FarmMes ANDs them into the per-channel
    current/rolling settings), yaw is never gated, and turb_TI adds one value.
    history_length is the history_N the caller writes into every channel dict
    (both the training script and the eval scripts override it from args/ckpt).
    """
    mes = config["mes_level"]
    H = int(history_length)

    def n(prefix: str, enabled: bool = True) -> int:
        c = config[f"{prefix}_mes"]
        return int(enabled) * (
            int(c[f"{prefix}_current"]) + int(c[f"{prefix}_rolling_mean"]) * H
        )

    return (
        n("ws", mes["turb_ws"])
        + n("wd", mes["turb_wd"])
        + n("yaw")
        + n("power", mes["turb_power"])
        + int(mes["turb_TI"])
    )
