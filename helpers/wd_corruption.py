"""Wind-direction corruption models for the T0 sensitivity injection (eval_wd.py).

T0 of the WD-estimation ladder asks: how much wind-direction error can the
trained policy tolerate before the ramp power_ratio degrades? To answer it we
corrupt ONLY the wd fed to the agent's rotation machinery (wind-relative
position transform + profile rotation) via the existing
``agent.act(..., wind_dirs=)`` override — the env physics, observations and
yaw re-booking always see the true wd. The largest corruption that stays
within the ramp noise floor is the error budget the T1 estimator must beat.

Spec strings (parsed by :func:`make_corruptor`):

``none``
    Identity (the uncorrupted control arm).
``bias:<deg>``
    Constant offset, e.g. ``bias:5`` or ``bias:-2``.
``ar1:sigma=<deg>,tau=<s>``
    Stationary AR(1) (Ornstein-Uhlenbeck in discrete time) additive noise:
    ``e[t+1] = rho * e[t] + sigma * sqrt(1 - rho^2) * xi``, with
    ``rho = exp(-dt_env / tau)`` and ``xi ~ N(0, 1)``. The ``sqrt(1 - rho^2)``
    innovation scaling makes ``sigma`` the STATIONARY standard deviation
    (independent of tau), so specs with different correlation times are
    directly comparable. Initialized at the stationary distribution.
``lag:<s>``
    Pure delay: the agent sees the true wd from ``<s>`` seconds ago
    (rounded to whole env steps). The delay line is pre-filled with the
    first true wd at reset, so early steps see the (correct) initial wd
    rather than garbage — matching a sensor that was already running
    during the burn-in.

All corruptors are stateful per vector-env slot and MUST be ``reset()`` on
episode boundaries; eval_wd.py resets them alongside ``env.reset()``.
"""
from __future__ import annotations

from collections import deque

import numpy as np


class WdCorruptor:
    """Base class: identity corruption (spec ``none``)."""

    def __init__(self, n_envs: int, dt_env: float):
        self.n_envs = int(n_envs)
        self.dt_env = float(dt_env)

    def reset(self, true_wd0: np.ndarray) -> None:
        """Start a new episode. ``true_wd0`` is the per-env wd at reset (deg)."""

    def corrupt(self, true_wd: np.ndarray) -> np.ndarray:
        """Map the per-env true wd (deg) to what the agent is fed (deg)."""
        return np.asarray(true_wd, dtype=np.float64).copy()


class BiasCorruptor(WdCorruptor):
    def __init__(self, n_envs: int, dt_env: float, bias_deg: float):
        super().__init__(n_envs, dt_env)
        self.bias_deg = float(bias_deg)

    def corrupt(self, true_wd: np.ndarray) -> np.ndarray:
        return np.asarray(true_wd, dtype=np.float64) + self.bias_deg


class Ar1Corruptor(WdCorruptor):
    def __init__(self, n_envs: int, dt_env: float, sigma_deg: float,
                 tau_s: float, seed: int = 0):
        super().__init__(n_envs, dt_env)
        if sigma_deg < 0 or tau_s <= 0:
            raise ValueError(f"ar1 needs sigma >= 0 and tau > 0, got "
                             f"sigma={sigma_deg}, tau={tau_s}")
        self.sigma_deg = float(sigma_deg)
        self.tau_s = float(tau_s)
        self.rho = float(np.exp(-self.dt_env / self.tau_s))
        self.rng = np.random.default_rng(seed)
        self._e = np.zeros(self.n_envs)

    def reset(self, true_wd0: np.ndarray) -> None:
        # Stationary start: no transient at the episode boundary.
        self._e = self.rng.normal(0.0, self.sigma_deg, size=self.n_envs)

    def corrupt(self, true_wd: np.ndarray) -> np.ndarray:
        out = np.asarray(true_wd, dtype=np.float64) + self._e
        innov = self.rng.normal(0.0, 1.0, size=self.n_envs)
        self._e = (self.rho * self._e
                   + self.sigma_deg * np.sqrt(1.0 - self.rho ** 2) * innov)
        return out


class LagCorruptor(WdCorruptor):
    def __init__(self, n_envs: int, dt_env: float, lag_s: float):
        super().__init__(n_envs, dt_env)
        if lag_s < 0:
            raise ValueError(f"lag must be >= 0, got {lag_s}")
        self.lag_s = float(lag_s)
        self.lag_steps = int(round(self.lag_s / self.dt_env))
        self._buf: deque = deque(maxlen=self.lag_steps + 1)

    def reset(self, true_wd0: np.ndarray) -> None:
        wd0 = np.asarray(true_wd0, dtype=np.float64).copy()
        self._buf.clear()
        for _ in range(self.lag_steps + 1):
            self._buf.append(wd0)

    def corrupt(self, true_wd: np.ndarray) -> np.ndarray:
        if len(self._buf) == 0:
            raise RuntimeError("LagCorruptor.corrupt() called before reset()")
        self._buf.append(np.asarray(true_wd, dtype=np.float64).copy())
        # maxlen = lag_steps + 1, so buf[0] is the value from lag_steps ago.
        return self._buf[0].copy()


def _parse_kv(body: str) -> dict:
    out = {}
    for part in body.split(","):
        part = part.strip()
        if not part:
            continue
        k, _, v = part.partition("=")
        if not _:
            raise ValueError(f"expected key=value, got {part!r}")
        out[k.strip()] = float(v)
    return out


def make_corruptor(spec: str, n_envs: int, dt_env: float,
                   seed: int = 0) -> WdCorruptor:
    """Parse a corruption spec string into a stateful corruptor.

    ``seed`` only matters for the stochastic (ar1) family; it should include
    the eval seed so paired specs across checkpoints share noise realizations.
    """
    spec = (spec or "none").strip()
    name, _, body = spec.partition(":")
    name = name.strip().lower()
    if name == "none":
        return WdCorruptor(n_envs, dt_env)
    if name == "bias":
        return BiasCorruptor(n_envs, dt_env, bias_deg=float(body))
    if name == "ar1":
        kv = _parse_kv(body)
        unknown = set(kv) - {"sigma", "tau"}
        if unknown:
            raise ValueError(f"ar1 spec has unknown keys {sorted(unknown)}")
        return Ar1Corruptor(n_envs, dt_env, sigma_deg=kv["sigma"],
                            tau_s=kv["tau"], seed=seed)
    if name == "lag":
        return LagCorruptor(n_envs, dt_env, lag_s=float(body))
    raise ValueError(f"Unknown wd corruption spec {spec!r} "
                     "(expected none | bias:<deg> | ar1:sigma=<deg>,tau=<s> "
                     "| lag:<s>)")
