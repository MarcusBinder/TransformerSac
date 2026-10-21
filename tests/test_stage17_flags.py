"""Stage-17 (exp_17) trainer flags: training-side wd quantisation
(``train_wd_quant_deg``) and the oracle-LUT reward-shaping curriculum
(``curriculum*``). Every flag defaults to the OLD behaviour, so the Stage-16
command lines are byte-identical in effect; the launcher's exact argv fragments
must parse through tyro.
"""
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import Args  # noqa: E402

CONFIG_PY = Path(__file__).resolve().parents[1] / "config.py"

# The launcher (paper-derating run_exp17_lumi.sh) fragments, verbatim.
CUR_RAMP = ("--curriculum lut --curriculum_w0 0 --curriculum_w1 1 --curriculum_start 50000 "
            "--curriculum_end 150000 --curriculum_sim_scale 0.1 --curriculum_ref_hold_deg 1.5")
CUR_RESID = CUR_RAMP.replace("--curriculum_w1 1 ", "--curriculum_w1 0.8 ")
NONORM = "--no-rotate_profiles --no-use_wind_relative_pos --wd_scaling_min 200 --wd_scaling_max 300"
LAUNCHER_FIELDS = ("train_wd_quant_deg", "curriculum", "curriculum_w0", "curriculum_w1", "curriculum_start",
                   "curriculum_end", "curriculum_sim_scale", "curriculum_ref_hold_deg")


def test_stage17_args_default_to_the_old_behaviour():
    a = Args()
    assert a.train_wd_quant_deg == 0.0
    assert a.curriculum == "none"
    assert a.curriculum_w0 == 0.0 and a.curriculum_w1 == 1.0
    assert a.curriculum_start == 50000 and a.curriculum_end == 150000
    assert a.curriculum_sim_scale == 0.1 and a.curriculum_sim_norm_deg == 45.0
    assert a.curriculum_ref_hold_deg == 1.5 and a.curriculum_ref_hold_center == 270.0
    assert a.curriculum_lut_ws == 9.5 and a.curriculum_lut_ti == 0.038 and a.curriculum_lut_sign == "farm"
    # Stage-16 neighbours untouched
    assert a.train_wd_function is None and a.wd_source == "true" and a.rotate_profiles is True
    assert a.use_wind_relative_pos is True and a.shuffle_turbs is True and a.obs_norm is False


def test_train_wd_quant_deg_parses_through_tyro():
    import tyro
    a = tyro.cli(Args, args=["--train_wd_function", "mix_static_richards", "--train_wd_quant_deg", "0.05"])
    assert a.train_wd_quant_deg == 0.05 and a.train_wd_function == "mix_static_richards"


def test_cur_ramp_fragment_parses_through_tyro():
    import tyro
    a = tyro.cli(Args, args=CUR_RAMP.split())
    assert a.curriculum == "lut" and a.curriculum_w0 == 0.0 and a.curriculum_w1 == 1.0
    assert a.curriculum_start == 50000 and a.curriculum_end == 150000
    assert a.curriculum_sim_scale == 0.1 and a.curriculum_ref_hold_deg == 1.5


def test_cur_resid_fragment_parses_through_tyro():
    import tyro
    a = tyro.cli(Args, args=CUR_RESID.split())
    assert a.curriculum == "lut" and a.curriculum_w1 == 0.8


def test_nonorm_fragment_parses_through_tyro():
    import tyro
    a = tyro.cli(Args, args=(NONORM + " --wd_source true --obs_agg raw15span --no-shuffle_turbs").split())
    assert a.rotate_profiles is False and a.use_wind_relative_pos is False
    assert a.wd_scaling_min == 200 and a.wd_scaling_max == 300
    assert a.wd_source == "true" and a.obs_agg == "raw15span" and a.shuffle_turbs is False


@pytest.mark.parametrize("field", LAUNCHER_FIELDS)
def test_every_launcher_flag_is_a_top_level_args_field_in_config_py(field):
    # the LUMI launcher's pin guard greps config.py for ``^    <field>:``
    assert re.search(rf"^    {field}:", CONFIG_PY.read_text(), re.M), field
