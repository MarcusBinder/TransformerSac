"""Reward-shaping wrappers for the optuna sweep (2026-10-22).

``YawReversalPenaltyWrapper`` closes the loop on the ``eval/yaw_reversal_rate``
diagnostic: a Change penalty charges every move, this one charges only moves
whose direction FLIPS relative to the previous move of the same turbine,
scaled by the size of the flipped move:

    r -= coef * mean_i[ 1(sign(d_t,i) * sign(d_{t-1,i}) < 0) * |d_t,i| / yaw_step_env ]

with ``d_t,i`` the physical yaw change of turbine i over the env step (read
from ``env.unwrapped.current_yaw``, so slew limits and the yaw bound are
already applied) and ``yaw_step_env`` the maximum move per step, so the
per-turbine term lies in [0, 1] and ``coef`` is comparable to
``--action_penalty``. The first move after a reset has no predecessor and
is never charged. Inserted in ``combined_wrapper`` BEFORE the ``--reward_scale``
TransformReward, like the Change penalty, so the scale multiplies it too.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np


class YawReversalPenaltyWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, coef: float):
        super().__init__(env)
        self.coef = float(coef)
        self._prev_yaw = None
        self._prev_delta = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_yaw = np.asarray(self.env.unwrapped.current_yaw, dtype=float).copy()
        self._prev_delta = None
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        base = self.env.unwrapped
        yaw = np.asarray(base.current_yaw, dtype=float)
        if self._prev_yaw is None:
            self._prev_yaw = yaw.copy()
            return obs, reward, terminated, truncated, info
        delta = yaw - self._prev_yaw
        penalty = 0.0
        if self._prev_delta is not None:
            flip = (np.sign(delta) * np.sign(self._prev_delta)) < 0
            step = float(base.yaw_step_env)
            penalty = self.coef * float(np.mean(flip * np.abs(delta) / step))
        self._prev_yaw = yaw.copy()
        self._prev_delta = delta
        info = dict(info)
        info["reversal_penalty"] = penalty
        return obs, reward - penalty, terminated, truncated, info


def maybe_wrap_reversal_penalty(env: gym.Env, coef: float) -> gym.Env:
    """Attach the wrapper only when ``coef > 0`` (0 = bitwise old reward)."""
    if coef is None or float(coef) <= 0.0:
        return env
    return YawReversalPenaltyWrapper(env, coef)
