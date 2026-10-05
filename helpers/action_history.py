"""Commanded-action history per turbine (Stage 10: adaptation-module input).

The policy's observation carries 600 s of REALISED yaw / power / ws / wd per
turbine but not the actions it commanded, so an actuator lag or dead time
(tau_yaw, delay_yaw) is not identifiable from the obs alone. This wrapper
keeps a ring of the last ``n_hist`` commanded delta-yaw actions per turbine
(PHYSICAL order, like every inner wrapper) and exposes them as ``n_feat``
evenly spaced samples spanning the ring, oldest -> newest, mirroring
``helpers.obs_agg._f_raw15span`` (so with n_hist 60 / n_feat 15 the history
covers the same 600 s window at the same 40 s spacing as the obs).

Zeros before the first command of an episode (padding, like the obs deque's
oldest-sample padding), cleared on reset. ``MultiLayoutEnv.action_history``
re-permutes the rows to token order and pads to max_turbines.
"""
from __future__ import annotations

from typing import Optional

import gymnasium as gym
import numpy as np

N_FEAT_DEFAULT = 15   # raw15span width (obs_agg): the adaptation module's per-turbine history width


def span_indices(n_hist: int, n_feat: int) -> np.ndarray:
    """Indices of n_feat evenly spaced samples over a chronological length-n_hist
    buffer, newest included (identical to obs_agg raw15span's linspace)."""
    return np.round(np.linspace(0, n_hist - 1, n_feat)).astype(int)


class ActionHistoryWrapper(gym.Wrapper):
    """Ring buffer of commanded yaw actions per turbine.

    Parameters
    ----------
    n_hist : int
        Ring length in env steps (60 = 600 s at dt_env 10 s).
    n_feat : int
        Samples exposed per turbine (15 = raw15span width).
    yaw_col : int
        Column of a 2-D action holding the yaw command; a flat action of
        ``n_turb`` entries is the yaw command itself, a longer flat action
        (``[yaw_0..yaw_n | derate...]``) uses its first ``n_turb`` entries.
    """

    def __init__(self, env: gym.Env, n_hist: int = 60, n_feat: int = 15, yaw_col: int = 0):
        super().__init__(env)
        if n_hist < 1 or n_feat < 1 or n_feat > n_hist:
            raise ValueError(f"need 1 <= n_feat <= n_hist, got n_feat={n_feat} n_hist={n_hist}")
        self.n_hist = int(n_hist)
        self.n_feat = int(n_feat)
        self.yaw_col = int(yaw_col)
        self.n_turb = int(getattr(env.unwrapped, "n_turb", 0) or getattr(env.unwrapped, "n_turbines", 0)
                          or int(np.prod(env.action_space.shape)))
        self._idx = span_indices(self.n_hist, self.n_feat)
        self._ring = np.zeros((self.n_hist, self.n_turb), dtype=np.float32)   # chronological
        self._count = 0

    # ------------------------------------------------------------------ api --
    @property
    def action_history(self) -> np.ndarray:
        """(n_turb, n_feat) float32, oldest -> newest, zeros where no command yet."""
        return self._ring[self._idx].T.copy()

    def _yaw_command(self, action) -> np.ndarray:
        a = np.asarray(action, dtype=np.float32)
        if a.ndim >= 2:
            a = a[..., self.yaw_col]
        a = a.reshape(-1)
        if a.shape[0] < self.n_turb:
            raise ValueError(f"action has {a.shape[0]} entries for {self.n_turb} turbines")
        return a[: self.n_turb]

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        self._ring[...] = 0.0
        self._count = 0
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        # Record the COMMAND before the env applies it (the obs after this step
        # reflects the realised response to it).
        self._ring[:-1] = self._ring[1:]
        self._ring[-1] = self._yaw_command(action)
        self._count += 1
        return self.env.step(action)
