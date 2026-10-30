"""helpers/tracking_env_config.py: the tracking trainer's env config layers.

The tracking trainer once built its config inline and skipped
``apply_config_overrides``, so ``--power_avg`` / ``--action_penalty`` /
``--power_scaling`` were parsed, saved in the checkpoint and dropped. These tests
pin the layer order through the real CLI and check that the trainer and the eval
rebuild both use the helper.
"""
import sys
from pathlib import Path

import tyro

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import Args  # noqa: E402
from helpers.env_configs import make_env_config  # noqa: E402
from helpers.tracking_env_config import build_tracking_env_config  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def parse(*argv):
    return tyro.cli(Args, args=["--config", "power_tracking_yaw", *argv])


def test_no_flags_is_the_preset_plus_action_method():
    args = parse()
    config, kwargs = build_tracking_env_config(args)
    expected = make_env_config("power_tracking_yaw")
    expected["ActionMethod"] = args.action_type
    assert config == expected
    assert kwargs == {}


def test_reward_flags_reach_the_config(capsys):
    args = parse("--power_avg", "1", "--power_scaling", "3.0",
                 "--action_penalty", "0.002", "--action_penalty_type", "change")
    config, _ = build_tracking_env_config(args)
    assert config["power_def"]["Power_avg"] == 1
    assert config["power_def"]["Power_scaling"] == 3.0
    assert config["act_pen"]["action_penalty"] == 0.002
    assert config["act_pen"]["action_penalty_type"] == "change"
    # The log line the cluster runs are checked against.
    assert "Config overrides: power_def.Power_avg: 5 -> 1" in capsys.readouterr().out


def test_env_overrides_win_over_the_reward_flags():
    args = parse("--power_avg", "1",
                 "--env_overrides", '{"power_def": {"Power_avg": 2}, "derate_max": 1.0}')
    config, _ = build_tracking_env_config(args)
    assert config["power_def"]["Power_avg"] == 2
    assert config["derate_max"] == 1.0


def test_turbine_model_layer():
    args = parse("--derate_reference", "setpoint", "--turbine_params", '{"tau_ct": 10}')
    config, kwargs = build_tracking_env_config(args)
    assert config["derate_reference"] == "setpoint"
    assert kwargs == {"dwm_params": {"tau_ct": 10.0}}


def test_checkpoint_args_dict_gives_the_same_config():
    args = parse("--power_avg", "1", "--action_penalty", "0.002",
                 "--env_overrides", '{"derate_max": 1.0}')
    assert build_tracking_env_config(vars(args)) == build_tracking_env_config(args)


def test_trainer_and_eval_rebuild_use_the_helper():
    for name in ("transformer_sac_windfarm_tracking.py", "eval_tracking_figs.py"):
        src = (REPO / name).read_text()
        assert "build_tracking_env_config(args)" in src, name
        assert "make_env_config(" not in src, name
