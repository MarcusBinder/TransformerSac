"""Residual-on-lut3 action prior (Optuna v3, plan 2026-10-29 WP3.5).

With ``--action_prior lut3 --action_prior_delta D`` the agent's per-turbine
action a in [-1, 1] is decoded as an ABSOLUTE yaw target

    target = lut3(phi_est) + a * D          (clipped to the yaw band)

where lut3 is the robust PyWake table (pywake_lut mode 3, sigma 2.3, sign farm)
built ONCE at the training point (midpoint of the training ws / TI bands, the
training turbine and layout, the checkpoint's yaw band) and indexed with the
env's own sensor-derived wd estimate ``env.wd_est`` in the agent's yaw frame
(``env.phi_frame`` under yaw_frame_source="est", else the live true frame,
exactly as the harness LutPolicy). a = 0 IS the harness lut3 row; the policy
can only add what RL learns on top (the dynamic gains). The decode is the
shared ``WindGym.core.yaw_frame.residual_target`` and the target is handed to
the env as its "wind" action (``wind_action_from_target``), so the rate limit
and the band are the env's. The DWM harness (EvalPretrainedAgentGif
TransformerPolicy) and the LES script apply the same ``prior_wind_action``.

Requires ActionMethod "wind" and an env built with an estimator
(--wd_source est); the yaw observation stays the absolute agent-frame yaw.
"""
from __future__ import annotations

from typing import Dict, Tuple

import gymnasium as gym
import numpy as np

from WindGym.core.yaw_frame import residual_target, wind_action_from_target

PRIOR_MODES = ("lut3",)
LUT_SIGMA = 2.3
LUT_SIGN = "farm"
_TABLE_CACHE: Dict[tuple, object] = {}


def _arg(args, name, default=None):
    if isinstance(args, dict):
        v = args.get(name, default)
    else:
        v = getattr(args, name, default)
    return default if v is None else v


def prior_point(args, config: dict) -> Tuple[float, float]:
    """(ws, ti) the prior table is built at: the midpoints of the TRAINING bands
    (--train_ws_min/max, --ti_min/max when given, else the preset's wind band).
    Pure function of the checkpoint args + preset, so the harness rebuilds the
    same table (rounded to 3 / 4 decimals)."""
    wind = config["wind"]
    ws_lo = float(_arg(args, "train_ws_min", wind["ws_min"]))
    ws_hi = float(_arg(args, "train_ws_max", wind["ws_max"]))
    ti_lo = float(_arg(args, "ti_min", wind["TI_min"]))
    ti_hi = float(_arg(args, "ti_max", wind["TI_max"]))
    return round(0.5 * (ws_lo + ws_hi), 3), round(0.5 * (ti_lo + ti_hi), 4)


def build_prior_controller(wind_turbine, x_pos, y_pos, ws: float, ti: float, yaw_max: float,
                           wd_nominal: float = 270.0, verbose: bool = False):
    """The lut3 controller (robust table) at the prior point; cached per process."""
    from helpers.pywake_lut import LutYawController, build_controller
    key = (type(wind_turbine).__name__, tuple(np.round(np.asarray(x_pos, float), 3)),
           tuple(np.round(np.asarray(y_pos, float), 3)), float(ws), float(ti), float(yaw_max))
    if key not in _TABLE_CACHE:
        _TABLE_CACHE[key] = build_controller(3, wind_turbine, x_pos, y_pos, ws=float(ws), ti=float(ti),
                                             yaw_max=float(yaw_max), sign=LUT_SIGN, sigma=LUT_SIGMA,
                                             wd_nominal=float(wd_nominal), verbose=verbose)
    base = _TABLE_CACHE[key]
    return LutYawController(3, None, base.lut, base.wd_grid, base.n_turb, wd_nominal=float(wd_nominal))


def lut_frame(base) -> float:
    """The frame the LUT target is expressed in = the agent's yaw frame (pywake_lut.LutPolicy)."""
    if getattr(base, "yaw_frame_source", "true") == "est":
        return float(base.phi_frame)
    return float(base.fs.wind_direction)


def prior_wind_action(a, ctrl, env, *, delta: float, yaw_min: float, yaw_max: float) -> np.ndarray:
    """The env "wind" action that commands ``lut3(phi_est) + a * delta`` on ``env``
    (a WindFarmEnv or anything exposing wd_est / phi_frame / yaw_frame_source / fs)."""
    base = getattr(env, "unwrapped", env)
    wd_hat = float(getattr(base, "wd_est", float("nan")))
    prior = np.asarray(ctrl.target_yaw(wd_hat, wd_frame=lut_frame(base)), dtype=float)
    target = residual_target(a, prior, delta, yaw_min, yaw_max)
    return wind_action_from_target(target, yaw_min, yaw_max).astype(np.float32)


class ActionPriorWrapper(gym.ActionWrapper):
    """Innermost action wrapper on a WindFarmEnv: residual action -> "wind" action."""

    def __init__(self, env: gym.Env, ctrl, delta: float):
        super().__init__(env)
        base = env.unwrapped
        if getattr(base, "ActionMethod", None) != "wind":
            raise ValueError("--action_prior needs ActionMethod 'wind' (absolute targets); "
                             f"the env has {getattr(base, 'ActionMethod', None)!r}")
        if getattr(base, "derate_action", False):
            raise ValueError("--action_prior is yaw-only (derate_action=True env)")
        if getattr(base, "wd_est_tau", None) is None:
            raise ValueError("--action_prior needs the env's wd_est (build it with wd_est_tau: "
                             "--wd_source est)")
        if float(getattr(ctrl, "deadband_deg", 0.0)) != 0.0:
            raise ValueError("the prior controller's deadband must be 0 (the estimator holds)")
        self.ctrl = ctrl
        self.delta = float(delta)
        self.yaw_min = float(base.yaw_min)
        self.yaw_max = float(base.yaw_max)

    def __getattr__(self, name):
        """Forward WindFarmEnv attributes the outer wrappers read directly (n_turb, x_pos,
        y_pos, D, farm_measurements, get_obs_dim_per_turbine, ...): gymnasium 1.x wrappers
        forward nothing, and PerTurbineObservationWrapper sits OUTSIDE this one."""
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self.env, name)

    def action(self, action):
        a = np.asarray(action, dtype=float).reshape(-1)[: self.unwrapped.n_turb]
        return prior_wind_action(a, self.ctrl, self.unwrapped, delta=self.delta,
                                 yaw_min=self.yaw_min, yaw_max=self.yaw_max)
