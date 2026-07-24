"""Named time-varying wind-direction schedules for evaluation (change_wd sweep).

A ``wd_function(t)`` returns the ABSOLUTE wind direction (deg) at simulation time
``t`` (seconds) and is passed straight to the pywake adapter via WindFarmEnv's
``wd_function`` kwarg. The env's burn-in holds the per-reset ``base_wd`` (see
wind_manager.py), so any config using one of these must pin
``wd_min = wd_max = wd_function(0)`` to avoid a wind-direction jump at t=0
(pattern: make_flow_gif.py).

Registered by name so SLURM scripts can select a schedule with a plain string
(``--eval_wd_function step_ramp_270_315``) instead of importing code.
"""
from __future__ import annotations

import numpy as np


def step_ramp_270_315(t):
    """The ``test_winddir_func`` schedule from test_changing_WD.ipynb, verbatim.

    270 -> 296.5 -> 315 deg: 300 s holds at each level, 200 s linear ramps between
    them. Full schedule completes at t = 1000 s.
    """
    # Goes from 270 -> 270+26.5 -> 270+45. It stays at each point for 100 steps.
    # Also moves over there in a linear fashion, so it is not a step function.
    t = np.asarray(t)

    wd_1 = 270
    wd_2 = 270 + 26.5
    wd_3 = 270 + 45

    time_at_wd = 300  # Time spent at each wind direction
    time_to_change = 200  # Time taken to change from one wind direction to the next

    wd = np.zeros_like(t, dtype=float)
    wd[t < time_at_wd] = wd_1
    wd[(t >= time_at_wd) & (t < time_at_wd + time_to_change)] = wd_1 + (wd_2 - wd_1) * ((t[(t >= time_at_wd) & (t < time_at_wd + time_to_change)] - time_at_wd) / time_to_change)
    wd[(t >= time_at_wd + time_to_change) & (t < 2 * time_at_wd + time_to_change)] = wd_2
    wd[(t >= 2 * time_at_wd + time_to_change) & (t < 2 * time_at_wd + 2 * time_to_change)] = wd_2 + (wd_3 - wd_2) * ((t[(t >= 2 * time_at_wd + time_to_change) & (t < 2 * time_at_wd + 2 * time_to_change)] - (2 * time_at_wd + time_to_change)) / time_to_change)
    wd[t >= 2 * time_at_wd + 2 * time_to_change] = wd_3

    return wd


def static_270(t):
    """Constant 270 deg — static-wd control matching step_ramp_270_315's start."""
    t = np.asarray(t)
    return np.full_like(t, 270.0, dtype=float)


WD_FUNCTIONS = {
    "step_ramp_270_315": step_ramp_270_315,
    "static_270": static_270,
}


def get_wd_function(name: str):
    """Look up a registered wd_function by name (ValueError lists known names)."""
    try:
        return WD_FUNCTIONS[name]
    except KeyError:
        raise ValueError(
            f"unknown wd_function '{name}' (known: {sorted(WD_FUNCTIONS)})"
        ) from None
