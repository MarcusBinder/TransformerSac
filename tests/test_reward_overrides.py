"""Tests for the change_wd_3 reward-conditioning / wind-range CLI overrides.

The change_wd_3 sweep asks what to do about the ~56 % of training states with no
wake-steering headroom (DTU10MW rates near 11.4 m/s while hard_2 draws
ws ~ U[10,14]): REWEIGHT them down by raising the Wake_recovery floor tau, or
STOP SAMPLING them by narrowing the training ws range. Its 15 arms therefore need
arbitrary combinations of (tau x Power_avg x Power_scaling x Power_reward x ws
range), which are supplied as None-defaulted CLI overrides rather than a dozen
near-duplicate ENV_CONFIGS presets.

Two invariants carry the whole sweep and are what this file guards:

1. Passing NO override flag must reproduce the preset exactly, or change_wd_3's
   anchor arm stops being comparable to change_wd_2's w2_tqc.
2. --train_ws_min / --train_ws_max are TRAINING-only. They are applied to the
   shared config, which the eval configs are deep-copied from, so the eval path's
   unconditional ws re-pin is the only thing keeping the training distribution out
   of the eval condition.
"""

import copy

import pytest
import tyro

from config import Args
from helpers.env_configs import (
    apply_config_overrides,
    make_env_config,
    make_eval_wind_config,
)
from helpers.wd_functions import build_eval_specs, get_wd_function

# A 2-turbine layout is enough to exercise reward construction; the reward
# calculator is layout-independent. 8D apart along x so the pywake backend has a
# real wake to compute.
X_POS = [0.0, 8 * 178.3]
Y_POS = [0.0, 0.0]


def parse(*argv):
    """Parse a change_wd_3-style command line through the real tyro CLI."""
    return tyro.cli(Args, args=list(argv))


def build_config(*argv, preset="hard_2"):
    args = parse(*argv)
    config = make_env_config(preset)
    apply_config_overrides(config, args)
    return args, config


# ---------------------------------------------------------------------------
# Defaults: every pre-change_wd_3 script must be unaffected
# ---------------------------------------------------------------------------

def test_all_override_flags_default_to_none():
    args = parse()
    for attr in ("reward_tau", "power_reward", "power_avg", "power_scaling",
                 "train_ws_min", "train_ws_max"):
        assert getattr(args, attr) is None, attr
    assert args.eval_ws == "12"


def test_no_flags_leaves_the_preset_untouched():
    _, overridden = build_config()
    assert overridden == make_env_config("hard_2")


@pytest.mark.parametrize("preset", ["hard", "hard_2", "hard_2_nowd", "default"])
def test_no_flags_is_a_no_op_for_every_preset_the_sweep_uses(preset):
    _, overridden = build_config(preset=preset)
    assert overridden == make_env_config(preset)


# ---------------------------------------------------------------------------
# Overrides land in the right config section
# ---------------------------------------------------------------------------

def test_reward_overrides_land_in_power_def():
    _, config = build_config(
        "--reward_tau", "0.10",
        "--power_avg", "1",
        "--power_reward", "Baseline",
        "--power_scaling", "0.2",
    )
    assert config["power_def"] == {
        "Power_reward": "Baseline",
        "Power_avg": 1,
        "Power_scaling": 0.2,
        "tau": 0.10,
    }


def test_tau_alone_leaves_the_rest_of_power_def_alone():
    """Arms 5-8 are single-factor: only the tau floor moves."""
    _, config = build_config("--reward_tau", "0.10")
    baseline = make_env_config("hard_2")["power_def"]
    assert config["power_def"]["Power_reward"] == baseline["Power_reward"]
    assert config["power_def"]["Power_avg"] == baseline["Power_avg"]
    assert config["power_def"]["Power_scaling"] == baseline["Power_scaling"]
    assert config["power_def"]["tau"] == 0.10


def test_train_ws_overrides_land_in_wind():
    _, config = build_config("--train_ws_min", "9.5", "--train_ws_max", "11.4")
    assert config["wind"]["ws_min"] == 9.5
    assert config["wind"]["ws_max"] == 11.4
    # wd domain randomization must survive -- these arms are still static-wd
    # over the full hard_2 band.
    assert (config["wind"]["wd_min"], config["wind"]["wd_max"]) == (225, 315)


# ---------------------------------------------------------------------------
# Train/eval separation: the ws override must not reach the evaluators
# ---------------------------------------------------------------------------

def test_eval_configs_are_pinned_despite_a_train_ws_override():
    _, train_config = build_config("--train_ws_min", "9.5", "--train_ws_max", "11.4")
    assert (train_config["wind"]["ws_min"], train_config["wind"]["ws_max"]) == (9.5, 11.4)

    specs = build_eval_specs("step_ramp_270_315,static_270,static_315", "12,10.5")
    for wd_name, ws, _ in specs:
        wd0 = float(get_wd_function(wd_name)(0.0))
        eval_config = make_eval_wind_config(train_config, wd0, ws)
        assert eval_config["wind"]["ws_min"] == ws
        assert eval_config["wind"]["ws_max"] == ws
        assert eval_config["wind"]["wd_min"] == wd0
        assert eval_config["wind"]["wd_max"] == wd0


def test_eval_configs_inherit_the_arms_reward_definition():
    """tau/Power_reward must NOT be re-pinned: the evaluator shares the arm's reward."""
    _, train_config = build_config("--reward_tau", "0.10", "--power_reward", "Baseline")
    eval_config = make_eval_wind_config(train_config, 270.0, 12.0)
    assert eval_config["power_def"] == train_config["power_def"]


def test_make_eval_wind_config_does_not_mutate_the_training_config():
    _, train_config = build_config("--train_ws_min", "10", "--train_ws_max", "12")
    snapshot = copy.deepcopy(train_config)
    make_eval_wind_config(train_config, 315.0, 10.5)
    assert train_config == snapshot


def test_static_315_pins_eval_wd_to_315():
    _, train_config = build_config()
    wd0 = float(get_wd_function("static_315")(0.0))
    eval_config = make_eval_wind_config(train_config, wd0, 12.0)
    assert eval_config["wind"]["wd_min"] == 315.0
    assert eval_config["wind"]["wd_max"] == 315.0


# ---------------------------------------------------------------------------
# End of the plumbing: the overrides reach the constructed env's RewardCalculator
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def wind_turbine():
    from py_wake.examples.data.dtu10mw import DTU10MW
    return DTU10MW()


def build_env(wind_turbine, config):
    from WindGym import WindFarmEnv
    return WindFarmEnv(
        x_pos=X_POS,
        y_pos=Y_POS,
        turbine=wind_turbine,
        config=config,
        backend="pywake",
        turbtype="Random",
        TurbBox="Default",
        dt_sim=10,
        dt_env=10,
        yaw_step_sim=5,
        n_passthrough=1,
        reset_init=False,
    )


def test_overrides_reach_the_reward_calculator(wind_turbine):
    _, config = build_config(
        "--reward_tau", "0.10",
        "--power_avg", "1",
        "--power_reward", "Baseline",
        "--power_scaling", "0.2",
    )
    env = build_env(wind_turbine, config)
    try:
        rc = env.reward_calculator
        assert rc.tau == 0.10
        assert rc.power_reward_type == "Baseline"
        assert rc._power_window_size == 1
        assert rc.power_scaling == 0.2
    finally:
        env.close()


def test_defaults_reach_the_reward_calculator_unchanged(wind_turbine):
    """No flags -> hard_2's reward, with tau at the env's 0.02 fallback."""
    _, config = build_config()
    env = build_env(wind_turbine, config)
    try:
        rc = env.reward_calculator
        assert rc.tau == 0.02
        assert rc.power_reward_type == "Wake_recovery"
        assert rc._power_window_size == 5
        assert rc.power_scaling == 1.0
    finally:
        env.close()


@pytest.mark.parametrize("tau", [0.05, 0.10, 0.20, 0.40])
def test_the_whole_tau_ladder_reaches_the_env(wind_turbine, tau):
    _, config = build_config("--reward_tau", str(tau))
    env = build_env(wind_turbine, config)
    try:
        assert env.reward_calculator.tau == tau
        # Single-factor: nothing else moved.
        assert env.reward_calculator.power_reward_type == "Wake_recovery"
        assert env.reward_calculator.power_scaling == 1.0
    finally:
        env.close()


def test_train_ws_override_reaches_the_env_wind_manager(wind_turbine):
    _, config = build_config("--train_ws_min", "10", "--train_ws_max", "12")
    env = build_env(wind_turbine, config)
    try:
        assert env.ws_inflow_min == 10
        assert env.ws_inflow_max == 12
    finally:
        env.close()
