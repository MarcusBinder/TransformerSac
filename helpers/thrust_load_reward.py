"""Thrust-based load terms for the tracking reward (no baseline farm).

The farm power-tracking literature prices loads with the agent farm's own
thrust, not with a greedy reference:

- ``dthrust2``: squared thrust increments per control step, summed over
  turbines (Zhao et al. 2017, IEEE TSTE, doi:10.1109/TSTE.2017.2673122;
  Zhong et al. 2023, EPSR, doi:10.1016/j.epsr.2023.109793, term S1;
  Boersma et al. 2019 as quoted by Zhong). Here the increments are taken at
  every dynamiks sim step (dt_sim) and summed over the env step's substeps,
  so cycling inside one agent step is seen; the first increment of a step is
  taken against the last sample of the previous step.
- ``thrust_eq``: squared deviation of each turbine's step-mean thrust from the
  farm mean (Zhong 2023, term S2; Vali et al. 2019, WES, load equalisation).
- ``thrust_level``: farm sum of step-mean thrust (static thrust-as-load, as in
  Soleimanzadeh et al. 2012 and the RL power-maximisation papers).
- ``thrust_max``: the worst turbine's thrust level, as a soft maximum
  ``(1/kappa) log sum_i exp(kappa z_i)`` of each turbine's low-passed thrust
  ``z_i`` (first-order, time constant ``tau``, updated every sim step,
  initialised at the first sample after reset). It lies between ``max_i z_i``
  and ``max_i z_i + log(N)/kappa``, and unlike the hard maximum it has a
  gradient for every turbine near the top.
- ``yaw_mis``: step mean of the farm-mean absolute yaw misalignment (deg, from
  the env's wind-relative ``info["yaws"]``) over ``YAW_REF_DEG`` (30 deg). Not
  a thrust term: it prices yaw offsets that the thrust terms reward (a yawed
  rotor has less axial inflow, hence less thrust).

All thrusts are T = 1/2 rho A u^2 Ct with u the rotor-averaged axial (rotor-
normal) wind speed and Ct from the turbine model, sampled after every sim
step by a dynamiks step handler, the same signal the powertracking eval
runner records, and normalised by ``thrust_ref`` (default: the turbine's peak
steady thrust over 3-25 m/s, from its own Ct curve). The penalty

    penalty = w_d * dthrust2 + w_e * thrust_eq + w_l * thrust_level
            + w_m * thrust_max + w_y * yaw_mis

is subtracted from the env reward before ``reward_scale``, like the DEL
penalty. Requires the dynamiks backend (``env.unwrapped.fs``).
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np

RHO = 1.225
YAW_REF_DEG = 30.0


class _ThrustProbe:
    """dynamiks step handler: Ct and rotor-averaged axial u after every sim step."""

    def __init__(self, fs):
        self.time, self.ct, self.u = [], [], []
        fs.add_step_handlers(self, t0=fs.time)

    def __call__(self, fs):
        wts = fs.windTurbines
        self.time.append(float(fs.time))
        self.ct.append(np.array(wts.ct(), dtype=float))
        self.u.append(np.array(wts.rotor_avg_windspeed[:, 0], dtype=float))

    def take(self):
        """(time, ct, u) since the last call, as arrays; the buffers are emptied."""
        out = np.array(self.time), np.array(self.ct), np.array(self.u)
        self.time, self.ct, self.u = [], [], []
        return out


def peak_thrust(turbine, rho: float = RHO) -> float:
    """Peak steady thrust (N) of a PyWake turbine over 3-25 m/s, undeflected and not derated."""
    ws = np.arange(3.0, 25.0 + 1e-9, 0.1)
    area = np.pi * float(turbine.diameter()) ** 2 / 4
    return float(np.max(0.5 * rho * area * ws**2 * np.asarray(turbine.ct(ws), dtype=float)))


def thrust_load_terms(thrust: np.ndarray, prev: np.ndarray | None) -> dict:
    """The three terms for one env step from normalised thrust (n_sub, n_turb).

    ``prev`` is the last normalised sample of the previous step (None on the
    first step after reset: the step's first sample is then its own reference).
    """
    t = np.asarray(thrust, dtype=float)
    ref = t[:1] if prev is None else np.asarray(prev, dtype=float)[None, :]
    d = np.diff(np.concatenate([ref, t]), axis=0)
    mean = t.mean(axis=0)
    return {
        "dthrust2": float(np.sum(d**2)),
        "thrust_eq": float(np.sum((mean - mean.mean()) ** 2)),
        "thrust_level": float(np.sum(mean)),
    }


def soft_max(z: np.ndarray, kappa: float) -> float:
    """``(1/kappa) log sum_i exp(kappa z_i)``: in ``[max z, max z + log(N)/kappa]``."""
    z = np.asarray(z, dtype=float)
    top = float(z.max())
    return top + float(np.log(np.sum(np.exp(kappa * (z - top))))) / kappa


def lowpass_thrust(thrust: np.ndarray, state: np.ndarray | None, *, dt: np.ndarray, tau: float):
    """First-order low-pass of normalised thrust (n_sub, n_turb) over one env step.

    ``z <- z + (1 - exp(-dt_j / tau)) (T_j - z)`` for every sim step ``j`` with
    its own ``dt_j`` (s). ``state`` is ``z`` after the previous step; ``None``
    (first step after reset) starts it at the first sample, whose ``dt`` is
    then unused. Returns ``z`` after the last sample.
    """
    t = np.asarray(thrust, dtype=float)
    start = 0
    if state is None:
        z, start = t[0].copy(), 1
    else:
        z = np.asarray(state, dtype=float).copy()
    for j in range(start, t.shape[0]):
        z += (1.0 - np.exp(-float(dt[j]) / tau)) * (t[j] - z)
    return z


class ThrustLoadRewardWrapper(gym.Wrapper):
    """reward -= w_d * dthrust2 + w_e * thrust_eq + w_l * thrust_level
    + w_m * thrust_max + w_y * yaw_mis (see module doc).

    info keys: ``load_penalty``, ``load_dthrust2``, ``load_thrust_eq``,
    ``load_thrust_level``, ``load_thrust_max``, ``load_yaw_mis`` (unweighted
    terms; ``load_yaw_mis`` only when the env reports ``info["yaws"]``) and
    ``reward_unpenalized``.
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        dthrust2_scale: float = 0.0,
        thrust_eq_scale: float = 0.0,
        thrust_level_scale: float = 0.0,
        thrust_max_scale: float = 0.0,
        thrust_max_kappa: float = 20.0,
        thrust_max_tau_s: float = 60.0,
        yaw_mis_scale: float = 0.0,
        thrust_ref: float | None = None,
    ) -> None:
        super().__init__(env)
        self.scales = {
            "dthrust2": float(dthrust2_scale),
            "thrust_eq": float(thrust_eq_scale),
            "thrust_level": float(thrust_level_scale),
            "thrust_max": float(thrust_max_scale),
            "yaw_mis": float(yaw_mis_scale),
        }
        if any(w < 0 for w in self.scales.values()):
            raise ValueError(f"load term scales must be >= 0, got {self.scales}")
        self.kappa, self.tau = float(thrust_max_kappa), float(thrust_max_tau_s)
        if self.kappa <= 0:
            raise ValueError(f"thrust_max_kappa must be > 0, got {thrust_max_kappa}")
        if self.tau <= 0:
            raise ValueError(f"thrust_max_tau_s must be > 0, got {thrust_max_tau_s}")
        base = env.unwrapped
        self.thrust_ref = float(thrust_ref) if thrust_ref else peak_thrust(base.turbine)
        self.area = np.pi * float(base.D) ** 2 / 4
        self._probe = None
        self._prev = None
        self._lp = None  # low-passed normalised thrust per turbine
        self._prev_time = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        fs = getattr(self.env.unwrapped, "fs", None)
        if fs is None or not hasattr(fs, "add_step_handlers"):
            raise RuntimeError("ThrustLoadRewardWrapper needs the dynamiks backend (env.unwrapped.fs)")
        # reset builds a new flow simulation, so the handler goes on the new one
        self._probe = _ThrustProbe(fs)
        self._prev = None
        self._lp = None
        self._prev_time = None
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        time, ct, u = self._probe.take()
        if "time_array" in info and len(time) != len(info["time_array"]):
            raise RuntimeError(
                f"thrust probe out of sync: {len(time)} samples vs "
                f"{len(info['time_array'])} sim steps in info['time_array']"
            )
        thrust = 0.5 * RHO * self.area * u**2 * ct / self.thrust_ref
        terms = thrust_load_terms(thrust, self._prev)
        self._prev = thrust[-1]
        dt = np.diff(time, prepend=np.nan if self._prev_time is None else self._prev_time)
        self._lp = lowpass_thrust(thrust, self._lp, dt=dt, tau=self.tau)
        self._prev_time = float(time[-1])
        terms["thrust_max"] = soft_max(self._lp, self.kappa)
        if "yaws" in info:
            yaws = np.asarray(info["yaws"], dtype=float)
            terms["yaw_mis"] = float(np.mean(np.abs(yaws))) / YAW_REF_DEG
        elif self.scales["yaw_mis"] > 0:
            raise RuntimeError("yaw_mis_scale > 0 needs info['yaws'] (per sim step, wind-relative)")
        penalty = sum(self.scales[k] * v for k, v in terms.items())
        info["reward_unpenalized"] = float(reward)
        info["load_penalty"] = float(penalty)
        for k, v in terms.items():
            info[f"load_{k}"] = v
        return obs, float(reward) - penalty, terminated, truncated, info
