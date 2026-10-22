"""Optuna-sweep trainer flags (LESRL optuna_sweep, 2026-10-22).

Every flag added for the sweep defaults to the OLD behaviour: a trainer run that
passes none of them must build a bit-identical env config / obs width and the
same reward. These tests guard that default-equality and the wiring of each
flag (A1-A7 of the sweep plan):

  A1 --wd_min/--wd_max/--yaw_min/--yaw_max  (config overrides, eval_wd too)
  A2 --target_entropy_scale                 (multiplies the adaptive target)
  A3 --eval_force_baseline                  (Baseline_comp on the eval envs)
  A4 --metrics_jsonl                        (append-only event log)
  A5 --obs_sensors / --obs_yaw_current      (sensor gating)
  A6 --reversal_penalty                     (YawReversalPenaltyWrapper)
  A7 hold_ramp_270_305_short                (eval registry)
"""
import json
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch
import tyro

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import Args  # noqa: E402
from helpers.env_configs import (  # noqa: E402
    apply_config_overrides,
    make_env_config,
    obs_dim_per_turbine,
)


def parse(*argv):
    return tyro.cli(Args, args=list(argv))


def build(*argv, preset="les_recipe"):
    args = parse(*argv)
    config = make_env_config(preset)
    apply_config_overrides(config, args)
    return args, config


# ---------------------------------------------------------------------------
# A1: wind-direction band + yaw band overrides
# ---------------------------------------------------------------------------

def test_a1_flags_default_to_none():
    a = Args()
    for attr in ("wd_min", "wd_max", "yaw_min", "yaw_max"):
        assert getattr(a, attr) is None, attr


def test_a1_no_flags_leaves_les_recipe_untouched():
    _, cfg = build()
    assert cfg == make_env_config("les_recipe")


def test_a1_wd_and_yaw_overrides_reach_the_config():
    _, cfg = build("--wd_min", "235", "--wd_max", "305", "--yaw_min", "-30", "--yaw_max", "30")
    assert cfg["wind"]["wd_min"] == 235 and cfg["wind"]["wd_max"] == 305
    assert cfg["farm"]["yaw_min"] == -30 and cfg["farm"]["yaw_max"] == 30


def test_a1_apply_config_overrides_accepts_a_checkpoint_args_dict():
    """eval_wd.py carries the checkpoint args as a dict, not an Args."""
    cfg = make_env_config("les_recipe")
    apply_config_overrides(cfg, {"yaw_min": -25.0, "yaw_max": 25.0, "wd_min": None})
    assert cfg["farm"] == {"yaw_min": -25.0, "yaw_max": 25.0}
    assert cfg["wind"]["wd_min"] == 225


def test_a1_eval_wd_build_eval_config_applies_the_overrides():
    from eval_wd import build_eval_config
    args = {**vars(Args()), "config": "les_recipe", "history_length": 15,
            "yaw_min": -30.0, "yaw_max": 30.0}
    cfg = build_eval_config(args, wd0=270.0, ws=9.0)
    assert cfg["farm"] == {"yaw_min": -30.0, "yaw_max": 30.0}
    assert cfg["wind"]["wd_min"] == cfg["wind"]["wd_max"] == 270.0
    assert cfg["wind"]["ws_min"] == cfg["wind"]["ws_max"] == 9.0


def test_a1_eval_wd_build_eval_config_default_is_the_old_construction():
    from eval_wd import build_eval_config
    from helpers.env_configs import make_eval_wind_config
    args = {**vars(Args()), "config": "les_recipe", "history_length": 15}
    cfg = build_eval_config(args, wd0=270.0, ws=9.0)
    ref = make_env_config("les_recipe")
    for mes_type, prefix in (("ws_mes", "ws"), ("wd_mes", "wd"), ("yaw_mes", "yaw"),
                             ("power_mes", "power")):
        ref[mes_type][f"{prefix}_history_N"] = 15
        ref[mes_type][f"{prefix}_history_length"] = 15
    assert cfg == make_eval_wind_config(ref, 270.0, 9.0)


# ---------------------------------------------------------------------------
# A2: target entropy scale
# ---------------------------------------------------------------------------

def test_a2_target_entropy_scale_defaults_to_one_and_multiplies():
    from helpers.training_utils import compute_adaptive_target_entropy
    assert Args().target_entropy_scale == 1.0
    mask = torch.tensor([[False, False, False, True], [False, True, True, True]])
    base = compute_adaptive_target_entropy(mask, 1, "sum")
    assert torch.equal(compute_adaptive_target_entropy(mask, 1, "sum", scale=1.0), base)
    assert torch.allclose(compute_adaptive_target_entropy(mask, 1, "sum", scale=0.5), 0.5 * base)
    assert torch.allclose(compute_adaptive_target_entropy(mask, 1, "mean", scale=2.0),
                          2.0 * compute_adaptive_target_entropy(mask, 1, "mean"))


# ---------------------------------------------------------------------------
# A3: force a baseline farm on the eval envs
# ---------------------------------------------------------------------------

def test_a3_eval_force_baseline_default_adds_no_kwargs():
    from helpers.env_configs import eval_env_extra_kwargs
    assert Args().eval_force_baseline is False
    assert eval_env_extra_kwargs(Args()) == {}
    assert eval_env_extra_kwargs(parse("--eval_force_baseline")) == {"Baseline_comp": True}


# ---------------------------------------------------------------------------
# A4: metrics jsonl
# ---------------------------------------------------------------------------

def test_a4_metrics_jsonl_defaults_to_none_and_is_a_noop():
    from helpers.metrics_jsonl import MetricsJsonl
    assert Args().metrics_jsonl is None
    m = MetricsJsonl(None)
    m.write("eval", step=1, **{"eval/wd/static_270/mean_power": 1.0})  # no error, no file


def test_a4_metrics_jsonl_appends_one_json_line_per_event(tmp_path):
    from helpers.metrics_jsonl import MetricsJsonl
    p = tmp_path / "sub" / "metrics.jsonl"
    m = MetricsJsonl(str(p))
    m.write("started", pid=123, obs_dim=4)
    m.write("eval", step=20000, **{"eval/wd/static_270/mean_power": 1.5e7})
    m.write("finished", step=20000, stop_requested=False)
    lines = p.read_text().splitlines()
    assert [json.loads(l)["event"] for l in lines] == ["started", "eval", "finished"]
    rec = json.loads(lines[1])
    assert rec["step"] == 20000 and rec["eval/wd/static_270/mean_power"] == 1.5e7
    assert "time" in rec
    # numpy scalars must serialise
    m.write("eval", step=np.int64(3), value=np.float32(1.25))
    assert json.loads(p.read_text().splitlines()[-1])["value"] == 1.25


# ---------------------------------------------------------------------------
# A5: sensor gating
# ---------------------------------------------------------------------------

def test_a5_sensor_flags_default_to_the_preset():
    from helpers.env_configs import apply_sensor_overrides
    a = Args()
    assert a.obs_sensors is None and a.obs_yaw_current is False
    cfg = make_env_config("les_recipe")
    apply_sensor_overrides(cfg, a)
    assert cfg == make_env_config("les_recipe")


@pytest.mark.parametrize("sensors, expect", [
    ("ws,wd,power", (True, True, True)),
    ("wd,power", (False, True, True)),
    ("ws", (True, False, False)),
    ("", (False, False, False)),
])
def test_a5_obs_sensors_gate_the_mes_level_flags(sensors, expect):
    from helpers.env_configs import apply_sensor_overrides
    args = parse("--obs_sensors", sensors)
    cfg = make_env_config("les_recipe")
    apply_sensor_overrides(cfg, args)
    mes = cfg["mes_level"]
    assert (mes["turb_ws"], mes["turb_wd"], mes["turb_power"]) == expect
    assert mes["turb_TI"] is False  # untouched
    # obs width follows: 15 rolling means per enabled channel + yaw
    assert obs_dim_per_turbine(cfg, 15) == 15 * (1 + sum(expect))


def test_a5_obs_sensors_rejects_unknown_names():
    from helpers.env_configs import apply_sensor_overrides
    with pytest.raises(ValueError):
        apply_sensor_overrides(make_env_config("les_recipe"), parse("--obs_sensors", "ws,ti"))


def test_a5_obs_yaw_current_adds_one_column():
    from helpers.env_configs import apply_sensor_overrides
    cfg = make_env_config("les_recipe")
    apply_sensor_overrides(cfg, parse("--obs_yaw_current"))
    assert cfg["yaw_mes"]["yaw_current"] is True
    assert obs_dim_per_turbine(cfg, 15) == 4 * 15 + 1


def test_a5_sensor_overrides_accept_a_dict():
    from helpers.env_configs import apply_sensor_overrides
    cfg = make_env_config("les_recipe")
    apply_sensor_overrides(cfg, {"obs_sensors": "wd", "obs_yaw_current": True})
    assert cfg["mes_level"]["turb_ws"] is False and cfg["yaw_mes"]["yaw_current"] is True


# ---------------------------------------------------------------------------
# A6: yaw reversal penalty
# ---------------------------------------------------------------------------

class _FakeYawEnv(gym.Env):
    """Minimal env exposing what the wrapper reads: current_yaw, yaw_step_env."""
    def __init__(self, traj):
        self.traj = [np.asarray(t, dtype=float) for t in traj]
        self.i = 0
        self.yaw_step_env = 5.0
        self.current_yaw = self.traj[0].copy()
        n = len(self.traj[0])
        self.observation_space = gym.spaces.Box(-1, 1, (n,), np.float32)
        self.action_space = gym.spaces.Box(-1, 1, (n,), np.float32)

    def reset(self, *, seed=None, options=None):
        self.i = 0
        self.current_yaw = self.traj[0].copy()
        return np.zeros(self.observation_space.shape, np.float32), {}

    def step(self, action):
        self.i += 1
        self.current_yaw = self.traj[self.i].copy()
        return np.zeros(self.observation_space.shape, np.float32), 1.0, False, False, {}


def test_a6_reversal_penalty_defaults_to_zero_and_zero_means_no_wrapper():
    from helpers.reward_variants import maybe_wrap_reversal_penalty
    assert Args().reversal_penalty == 0.0
    env = _FakeYawEnv([[0, 0]])
    assert maybe_wrap_reversal_penalty(env, 0.0) is env
    assert maybe_wrap_reversal_penalty(env, 0.1) is not env


def test_a6_reversal_penalty_charges_sign_flips_scaled_by_step_size():
    from helpers.reward_variants import YawReversalPenaltyWrapper
    # turbine 0: +5, -2.5, -2.5 (one flip of 2.5 deg); turbine 1: +5, +5, 0 (none)
    traj = [[0, 0], [5, 5], [2.5, 10], [0, 10]]
    env = YawReversalPenaltyWrapper(_FakeYawEnv(traj), coef=2.0)
    env.reset()
    _, r1, *_ = env.step(None)   # first move: no previous delta -> no penalty
    _, r2, *_ = env.step(None)   # t0 flips (+5 -> -2.5): |d|/step = 0.5, mean over 2 turbines = 0.25
    _, r3, *_ = env.step(None)   # t0 continues (-2.5 -> -2.5): no flip; t1 stops: no flip
    assert r1 == 1.0
    assert r2 == pytest.approx(1.0 - 2.0 * 0.25)
    assert r3 == 1.0
    env.reset()
    _, r1b, *_ = env.step(None)
    assert r1b == 1.0  # history cleared on reset


def test_a6_reversal_penalty_sits_before_reward_scale():
    """combined_wrapper inserts it before TransformReward, so --reward_scale
    multiplies the penalised reward: check the composition order used there."""
    from helpers.reward_variants import YawReversalPenaltyWrapper
    traj = [[0], [5], [0]]
    env = YawReversalPenaltyWrapper(_FakeYawEnv(traj), coef=1.0)
    env = gym.wrappers.TransformReward(env, lambda r: r * 10.0)
    env.reset()
    env.step(None)
    _, r2, *_ = env.step(None)
    assert r2 == pytest.approx(10.0 * (1.0 - 1.0))


# ---------------------------------------------------------------------------
# A7: eval registry
# ---------------------------------------------------------------------------

def test_a7_hold_ramp_270_305_short_mirrors_the_235_case():
    from helpers.wd_functions import WD_FUNCTIONS
    f = WD_FUNCTIONS["hold_ramp_270_305_short"]
    g = WD_FUNCTIONS["hold_ramp_270_235_short"]
    t = np.array([0.0, 299.0, 300.0, 600.0, 900.0, 2240.0])
    assert np.allclose(f(t), [270, 270, 270, 287.5, 305, 305])
    assert np.allclose(f(t) - 270.0, -(g(t) - 270.0))


# ---------------------------------------------------------------------------
# --num_eval_envs: decouple the eval vector-env size from the training one
# ---------------------------------------------------------------------------

def test_num_eval_envs_defaults_to_the_training_num_envs():
    from helpers.env_configs import eval_num_envs
    assert Args().num_eval_envs is None
    assert eval_num_envs(parse("--num_envs", "30")) == 30
    assert eval_num_envs(parse("--num_envs", "30", "--num_eval_envs", "8")) == 8
    with pytest.raises(ValueError):
        eval_num_envs(parse("--num_eval_envs", "0"))
