"""Integration tests for the --wd_source est plumbing (WD ladder, Phase 6).

Two contracts:

  * ``wd_est`` is reachable through the full training env stack
    (WindFarmEnv -> MultiLayoutEnv -> AsyncVectorEnv ``get_attr``), returns
    finite degrees, and is NOT the privileged ``wd`` scalar (on a turbulent
    dynamiks flow the sensor-derived estimate differs from the global truth).
  * Buffer honesty: transitions stored via the est-sourced fetch put the
    ESTIMATE into ReplayBuffer._wind_directions, so gradients trained from
    the buffer can never see the true wd.

The dynamiks episode is deliberately tiny (2 turbines, 3 env steps) — this
is a seam test, not a physics test.
"""
import os
import sys

import numpy as np
import pytest
import gymnasium as gym

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from helpers.helper_funcs import get_env_wind_directions  # noqa: E402
from helpers.multi_layout_env import MultiLayoutEnv, LayoutConfig  # noqa: E402
from replay_buffer import TransformerReplayBuffer  # noqa: E402


N_STEPS = 3
DT = 10


@pytest.fixture(scope="module")
def vec_env_with_estimator():
    from py_wake.examples.data.dtu10mw import DTU10MW
    from WindGym import WindFarmEnv
    from WindGym.wrappers import PerTurbineObservationWrapper
    from helpers.env_configs import make_env_config
    from helpers.layouts import get_layout_positions

    turbine = DTU10MW()
    x_pos, y_pos = get_layout_positions("test_layout", turbine)
    layout = LayoutConfig(name="test_layout", x_pos=x_pos, y_pos=y_pos)

    config = make_env_config("hard_2")
    config["wind"]["ws_min"] = config["wind"]["ws_max"] = 10.0
    config["wind"]["wd_min"] = config["wind"]["wd_max"] = 270.0

    def env_factory(x, y):
        return WindFarmEnv(
            turbine=turbine, x_pos=x, y_pos=y, config=config,
            backend="dynamiks", turbtype="MannGenerate",
            dt_sim=DT, dt_env=DT, reset_init=False,
            wd_est_tau=30.0, wd_est_consensus="median",
        )

    def make_env_fn(seed):
        def _init():
            return MultiLayoutEnv(
                layouts=[layout], env_factory=env_factory,
                per_turbine_wrapper=PerTurbineObservationWrapper,
                seed=seed, shuffle=False, max_episode_steps=50,
            )
        return _init

    envs = gym.vector.AsyncVectorEnv(
        [make_env_fn(1)], autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )

    class _Shim:  # the trainer fetches through envs.env.get_attr
        def __init__(self, env):
            self.env = env

    shim = _Shim(envs)
    obs, infos = envs.reset(seed=1)
    yield shim, envs, obs, infos
    envs.close()


@pytest.mark.slow
def test_wd_est_reachable_and_distinct(vec_env_with_estimator):
    shim, envs, obs, infos = vec_env_with_estimator

    est = get_env_wind_directions(shim, attr="wd_est")
    true = get_env_wind_directions(shim, attr="wd")
    assert est.shape == true.shape == (1,)
    assert np.isfinite(est).all(), "estimator must be warm after reset"
    assert 0.0 <= est[0] < 360.0

    # Across a few turbulent steps the sensor-derived estimate must not be
    # bit-identical to the privileged global scalar at every step.
    diffs = [abs(float(est[0]) - float(true[0]))]
    for _ in range(N_STEPS):
        actions = envs.action_space.sample() * 0.0
        obs, r, term, trunc, infos = envs.step(actions)
        est = get_env_wind_directions(shim, attr="wd_est")
        true = get_env_wind_directions(shim, attr="wd")
        assert np.isfinite(est).all()
        diffs.append(abs(float(est[0]) - float(true[0])))
    assert max(diffs) > 1e-6, (
        "wd_est identical to wd on a turbulent flow — estimator is "
        "probably reading the privileged scalar")
    # ...but it must still be a sane estimate of the true direction.
    assert max(diffs) < 45.0

    # The estimate must also surface in the info dict.
    assert "Wind direction estimate" in infos


@pytest.mark.slow
def test_buffer_honesty(vec_env_with_estimator):
    """rb._wind_directions stores the est trace, not the true wd trace."""
    shim, envs, obs, infos = vec_env_with_estimator
    n_turb = obs.shape[1]

    rb = TransformerReplayBuffer(
        capacity=16, device="cpu", rotor_diameter=178.3,
        max_turbines=n_turb, obs_dim=obs.shape[-1], action_dim=1,
    )

    est_trace, true_trace = [], []
    for _ in range(N_STEPS):
        wind_dirs = get_env_wind_directions(shim, attr="wd_est")  # est path
        true_trace.append(float(
            get_env_wind_directions(shim, attr="wd")[0]))
        est_trace.append(float(wind_dirs[0]))
        actions = envs.action_space.sample() * 0.0
        next_obs, rewards, terms, truncs, infos = envs.step(actions)
        rb.add(
            obs[0], next_obs[0],
            actions[0].reshape(-1, 1), rewards[0],
            bool(terms[0] or truncs[0]),
            np.zeros((n_turb, 2), dtype=np.float32),
            np.zeros(n_turb, dtype=bool),
            wind_dirs[0],
        )
        obs = next_obs

    stored = rb._wind_directions[:N_STEPS].astype(float)
    np.testing.assert_allclose(stored, est_trace, rtol=1e-6)
    assert not np.allclose(stored, true_trace), (
        "buffer stored the TRUE wd trace under wd_source=est — honesty "
        "violation, gradients would see the privileged signal")
