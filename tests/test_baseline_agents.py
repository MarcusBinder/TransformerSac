"""helpers/baseline_agents.py: duck-typed PolicyEvaluator agents (LUT / greedy)
for ``--baseline_eval_policy`` (optuna sweep, 2026-10-22).

The LUT agent must produce, through the evaluator's vector-env path
(``get_attr`` on a MultiLayoutEnv), exactly the action ``LutPolicy.predict``
produces on the underlying WindFarmEnv -- same controller, same action math --
so the sweep's reference is the harness LUT measured in the trainer's own eval
envs. A synthetic table keeps the test off the (slow) PyWake optimiser.
"""
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers.env_configs import make_env_config  # noqa: E402
from helpers.multi_layout_env import LayoutConfig, MultiLayoutEnv  # noqa: E402
from helpers.pywake_lut import LutPolicy, LutYawController  # noqa: E402

X_POS = [0.0, 8 * 178.3]
Y_POS = [0.0, 0.0]
MAX_TURBINES = 3  # one padding slot, so the action must be padded too


@pytest.fixture(scope="module")
def wind_turbine():
    from py_wake.examples.data.dtu10mw import DTU10MW
    return DTU10MW()


def _config(action_method):
    cfg = make_env_config("les_recipe")
    cfg["ActionMethod"] = action_method
    for p in ("ws", "wd", "yaw", "power"):
        cfg[f"{p}_mes"][f"{p}_history_N"] = 3
        cfg[f"{p}_mes"][f"{p}_history_length"] = 3
        cfg[f"{p}_mes"][f"{p}_window_length"] = 1
    return cfg


def _vec_env(wind_turbine, action_method, yaw_init="Random"):
    from WindGym import WindFarmEnv
    from WindGym.wrappers import PerTurbineObservationWrapper
    cfg = _config(action_method)
    cfg["yaw_init"] = yaw_init

    def factory(x_pos, y_pos):
        return WindFarmEnv(x_pos=x_pos, y_pos=y_pos, turbine=wind_turbine, config=cfg,
                           backend="pywake", turbtype="Random", TurbBox="Default",
                           dt_sim=10, dt_env=10, yaw_step_sim=0.5, n_passthrough=1,
                           reset_init=False)

    def make():
        return MultiLayoutEnv(layouts=[LayoutConfig(name="two", x_pos=np.array(X_POS), y_pos=np.array(Y_POS))],
                              env_factory=factory, per_turbine_wrapper=PerTurbineObservationWrapper,
                              seed=3, shuffle=False, max_turbines=MAX_TURBINES, max_episode_steps=50)
    return gym.vector.SyncVectorEnv([make]), cfg


def _synthetic_controller():
    wd_grid = np.arange(225.0, 316.0, 5.0)
    # offsets: turbine 0 steers with (wd - 270) / 2, turbine 1 stays greedy
    lut = np.stack([(wd_grid - 270.0) / 2.0, np.zeros_like(wd_grid)], axis=1)
    return LutYawController(3, lut, lut, wd_grid, 2, wd_nominal=270.0)


def _base_env(envs):
    return envs.envs[0]._get_base_env()


@pytest.mark.parametrize("action_method", ["yaw", "wind"])
def test_lut_eval_agent_matches_lut_policy_predict(wind_turbine, action_method):
    from helpers.baseline_agents import LutEvalAgent
    envs, cfg = _vec_env(wind_turbine, action_method)
    try:
        obs, _ = envs.reset(seed=11)
        base = _base_env(envs)
        base.wd = 262.0            # off-nominal so the table row matters
        base.fs.wind_direction = 262.0
        ctrl = _synthetic_controller()
        agent = LutEvalAgent(ctrl, yaw_min=cfg["farm"]["yaw_min"], yaw_max=cfg["farm"]["yaw_max"])
        agent.eval()
        act = agent.act(envs, obs, deterministic=True)
        ref, _ = LutPolicy(_synthetic_controller(), base, cfg["farm"]["yaw_max"],
                           cfg["farm"]["yaw_min"]).predict()
        assert act.shape == (1, MAX_TURBINES) and act.dtype == np.float32
        assert np.allclose(act[0, :2], ref)
        assert act[0, 2] == 0.0  # padding slot
        agent.train()
    finally:
        envs.close()


def test_greedy_eval_agent_drives_yaw_to_zero(wind_turbine):
    from helpers.baseline_agents import GreedyEvalAgent
    envs, cfg = _vec_env(wind_turbine, "yaw")
    try:
        obs, _ = envs.reset(seed=5)
        base = _base_env(envs)
        base.current_yaw = np.array([2.0, -20.0])
        agent = GreedyEvalAgent(yaw_min=cfg["farm"]["yaw_min"], yaw_max=cfg["farm"]["yaw_max"])
        act = agent.act(envs, obs, deterministic=True)
        step = float(base.yaw_step_env)
        assert np.allclose(act[0, :2], np.clip(-base.current_yaw / step, -1, 1))
    finally:
        envs.close()


def test_lut_eval_agent_rejects_shuffled_envs(wind_turbine):
    from helpers.baseline_agents import LutEvalAgent
    envs, cfg = _vec_env(wind_turbine, "yaw")
    try:
        obs, _ = envs.reset(seed=5)
        envs.envs[0].shuffle = True
        envs.envs[0]._perm = np.array([1, 0])
        agent = LutEvalAgent(_synthetic_controller(), yaw_min=-45, yaw_max=45)
        with pytest.raises(RuntimeError):
            agent.act(envs, obs)
    finally:
        envs.close()


def test_baseline_eval_policy_flag_parses_and_defaults_to_none():
    import tyro
    from config import Args
    assert Args().baseline_eval_policy is None
    assert tyro.cli(Args, args=["--baseline_eval_policy", "lut3"]).baseline_eval_policy == "lut3"
