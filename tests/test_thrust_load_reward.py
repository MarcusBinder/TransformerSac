"""helpers/thrust_load_reward.py: thrust-based load terms for the tracking reward.

Hand-computable cases for the three terms, and the wrapper on a fake dynamiks
env (step handlers fired once per sim step, a new flow simulation per reset).
"""
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers.thrust_load_reward import (  # noqa: E402
    RHO,
    ThrustLoadRewardWrapper,
    peak_thrust,
    thrust_load_terms,
)


def test_terms_hand_computed():
    # 2 substeps x 2 turbines; previous step ended at (0.5, 0.3)
    t = np.array([[0.6, 0.3], [0.4, 0.5]])
    terms = thrust_load_terms(t, prev=np.array([0.5, 0.3]))
    # increments: (0.1, 0), (-0.2, 0.2) -> 0.01 + 0.04 + 0.04
    assert terms["dthrust2"] == pytest.approx(0.09)
    # step means (0.5, 0.4), farm mean 0.45 -> 2 * 0.05^2
    assert terms["thrust_eq"] == pytest.approx(0.005)
    assert terms["thrust_level"] == pytest.approx(0.9)


def test_first_step_has_no_jump_from_nothing():
    t = np.array([[0.6, 0.3], [0.4, 0.5]])
    # only the within-step increment (-0.2, 0.2)
    assert thrust_load_terms(t, prev=None)["dthrust2"] == pytest.approx(0.08)


class _Turbine:
    def diameter(self):
        return 2.0

    def ct(self, ws):
        return np.where(np.asarray(ws) < 10.0, 0.8, 0.8 * 100.0 / np.asarray(ws) ** 2)


class _WTs:
    def __init__(self, sim):
        self.sim = sim

    def ct(self):
        return self.sim.ct_now

    @property
    def rotor_avg_windspeed(self):
        return np.stack([self.sim.u_now, np.zeros_like(self.sim.u_now)], axis=1)


class _FS:
    def __init__(self):
        self.time = 0.0
        self.handlers = []
        self.windTurbines = _WTs(self)
        self.ct_now = np.zeros(2)
        self.u_now = np.zeros(2)

    def add_step_handlers(self, handler, t0=0, dt=None):
        self.handlers.append(handler)

    def sim_step(self, ct, u):
        self.time += 1.0
        self.ct_now, self.u_now = np.asarray(ct, float), np.asarray(u, float)
        for h in self.handlers:
            h(self)


class _Env(gym.Env):
    """Two turbines, two sim steps per env step, reward -1, scripted (Ct, u)."""

    observation_space = gym.spaces.Box(-1, 1, (1,))
    action_space = gym.spaces.Box(-1, 1, (1,))

    def __init__(self, script):
        self.turbine = _Turbine()
        self.D = 2.0
        self.fs = None
        self.script = script

    def reset(self, seed=None, options=None):
        self.fs = _FS()  # a fresh flow simulation per episode, as in WindGym
        self.fs.sim_step([0.9, 0.9], [9.0, 9.0])  # warm-up: before the probe is attached
        self.k = 0
        return np.zeros(1, np.float32), {}

    def step(self, action):
        times = []
        for ct, u in self.script[self.k]:
            self.fs.sim_step(ct, u)
            times.append(self.fs.time)
        self.k += 1
        return np.zeros(1, np.float32), -1.0, False, False, {"time_array": np.array(times)}


def _thrust(ct, u, d=2.0):
    return 0.5 * RHO * np.pi * d**2 / 4 * u**2 * ct


def test_peak_thrust_of_fake_turbine():
    # Ct = 0.8 below 10 m/s and 80/u^2 above: thrust grows to 10 m/s, then stays flat
    assert peak_thrust(_Turbine()) == pytest.approx(_thrust(0.8, 10.0))


def test_wrapper_penalty_and_info():
    tref = _thrust(0.8, 10.0)
    script = [
        [([0.8, 0.8], [10.0, 5.0]), ([0.8, 0.8], [10.0, 5.0])],
        [([0.8, 0.8], [5.0, 5.0]), ([0.8, 0.8], [5.0, 10.0])],
    ]
    env = ThrustLoadRewardWrapper(
        _Env(script), dthrust2_scale=2.0, thrust_eq_scale=3.0, thrust_level_scale=0.5
    )
    env.reset()
    _, r1, *_, info1 = env.step(None)
    # normalised thrust: 1 at 10 m/s, 0.25 at 5 m/s; the warm-up sample is not seen
    assert info1["load_dthrust2"] == pytest.approx(0.0)
    assert info1["load_thrust_eq"] == pytest.approx(2 * 0.375**2)
    assert info1["load_thrust_level"] == pytest.approx(1.25)
    assert env.thrust_ref == pytest.approx(tref)
    pen1 = 3.0 * 2 * 0.375**2 + 0.5 * 1.25
    assert info1["load_penalty"] == pytest.approx(pen1)
    assert r1 == pytest.approx(-1.0 - pen1)
    assert info1["reward_unpenalized"] == -1.0

    _, r2, *_, info2 = env.step(None)
    # from the previous step's last sample (1, 0.25): (-0.75, 0), then (0, 0.75)
    assert info2["load_dthrust2"] == pytest.approx(2 * 0.75**2)

    # a new episode starts without a jump from the old one's last sample
    env.reset()
    _, _, *_, info3 = env.step(None)
    assert info3["load_dthrust2"] == pytest.approx(0.0)


def test_explicit_thrust_ref_and_negative_scale():
    env = ThrustLoadRewardWrapper(_Env([]), dthrust2_scale=1.0, thrust_ref=2.0e6)
    assert env.thrust_ref == 2.0e6
    with pytest.raises(ValueError, match=">= 0"):
        ThrustLoadRewardWrapper(_Env([]), dthrust2_scale=-1.0)


def test_needs_dynamiks_fs():
    class _NoFS(_Env):
        def reset(self, seed=None, options=None):
            return np.zeros(1, np.float32), {}

    env = ThrustLoadRewardWrapper(_NoFS([]), dthrust2_scale=1.0)
    with pytest.raises(RuntimeError, match="dynamiks"):
        env.reset()
