"""helpers/turbine_model.py: --derate_reference / --turbine_params -> env config + kwargs.

``derate_reference`` overrides the preset's key (None keeps the preset);
``turbine_params`` is a JSON object of WindGym turbine-parameter keys that
becomes the ``dwm_params`` constructor kwarg (empty -> no kwarg, so the
nominal path is untouched).
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers.turbine_model import apply_turbine_model, turbine_model_spec  # noqa: E402


def test_none_and_empty_leave_config_and_kwargs_alone():
    config = {"derate_reference": "available", "x": 1}
    kwargs = {"backend": "dynamiks"}
    apply_turbine_model(config, kwargs, derate_reference=None, turbine_params="{}")
    assert config == {"derate_reference": "available", "x": 1}
    assert kwargs == {"backend": "dynamiks"}


def test_override_sets_reference_and_dwm_params():
    config = {"derate_reference": "available"}
    kwargs = {}
    apply_turbine_model(
        config, kwargs, derate_reference="setpoint",
        turbine_params='{"tau_power": 6, "tau_ct": 10.0}',
    )
    assert config["derate_reference"] == "setpoint"
    assert kwargs["dwm_params"] == {"tau_power": 6.0, "tau_ct": 10.0}
    assert all(isinstance(v, float) for v in kwargs["dwm_params"].values())


def test_turbine_params_merge_into_existing_dwm_params():
    kwargs = {"dwm_params": {"k1": 0.1}}
    apply_turbine_model({}, kwargs, derate_reference=None, turbine_params='{"tau_ct": 10}')
    assert kwargs["dwm_params"] == {"k1": 0.1, "tau_ct": 10.0}


@pytest.mark.parametrize("bad", ["[1, 2]", '{"tau_ct": "ten"}', "not json"])
def test_bad_turbine_params_raise(bad):
    with pytest.raises(ValueError, match="turbine_params"):
        apply_turbine_model({}, {}, derate_reference=None, turbine_params=bad)


def test_spec_records_both_fields_from_saved_args():
    assert turbine_model_spec({"derate_reference": "setpoint", "turbine_params": '{"tau_ct": 10}'}) == {
        "derate_reference": "setpoint", "turbine_params": {"tau_ct": 10.0},
    }
    # checkpoints saved before the flags existed resolve to their preset's meaning
    assert turbine_model_spec({}) == {"derate_reference": "available", "turbine_params": {}}


# ------------------------------------------------- resolved spec and the guard --
def test_spec_resolves_a_missing_flag_from_the_preset():
    # a checkpoint trained without --derate_reference ran on the preset's meaning
    spec = turbine_model_spec({"config": "power_tracking_yaw"})
    assert spec["derate_reference"] == "available" and spec["turbine_params"] == {}
    assert turbine_model_spec({"config": "power_tracking_yaw", "derate_reference": "setpoint"})[
        "derate_reference"
    ] == "setpoint"


def test_check_turbine_model_refuses_mismatches():
    from types import SimpleNamespace

    from helpers.turbine_model import check_turbine_model

    spec = {"derate_reference": "setpoint", "turbine_params": {"tau_power": 6.0, "tau_ct": 10.0}}
    ok = SimpleNamespace(derate_reference="setpoint", tau_power=6.0, tau_ct=10.0)
    check_turbine_model(spec, ok)
    check_turbine_model(spec, SimpleNamespace())  # facts without the fields: nothing to compare
    with pytest.raises(ValueError, match="derate_reference"):
        check_turbine_model(spec, SimpleNamespace(derate_reference="available"))
    with pytest.raises(ValueError, match="tau_ct"):
        check_turbine_model(spec, SimpleNamespace(derate_reference="setpoint", tau_power=6.0, tau_ct=0.0))
    # a legacy checkpoint (no turbine params) evaluated on a nominal turbine is fine
    check_turbine_model(
        {"derate_reference": "available", "turbine_params": {}},
        SimpleNamespace(derate_reference="available", tau_power=0.0, tau_ct=0.0),
    )
    # the derate action range is part of what the policy learned
    spec_r = {**spec, "derate_min": 0.0, "derate_max": 0.8}
    check_turbine_model(spec_r, SimpleNamespace(derate_reference="setpoint", derate_max=0.8))
    with pytest.raises(ValueError, match="derate_max"):
        check_turbine_model(spec_r, SimpleNamespace(derate_reference="setpoint", derate_max=1.0))
    with pytest.raises(ValueError, match="derate_min"):
        check_turbine_model(spec_r, SimpleNamespace(derate_min=0.1))


def test_build_track3_env_carries_the_checkpoint_turbine_model(monkeypatch):
    """The probe env and the rollout env both get the checkpoint's reference and dwm_params."""
    import eval_tracking_figs as etf

    seen = {}

    def fake_measure_greedy(make_env):
        seen["probe"] = make_env()
        return 1.0e7

    monkeypatch.setattr(etf, "measure_greedy", fake_measure_greedy)
    monkeypatch.setattr(etf, "make_operating_point_lookup_for", lambda *a, **k: None)
    args = {
        "turbtype": "DTU10MW", "iea34_variant": "annrpm", "config": "power_tracking_yaw",
        "action_type": "wind", "max_eps": 1, "TI_type": "None", "backend": "dynamiks",
        "dt_sim": 1, "dt_env": 1, "yaw_step": 0.5, "use_wd_deviation": False,
        "wd_scale_range": 90.0, "power_schedule": "default",
        "derate_reference": "setpoint", "turbine_params": '{"tau_power": 6, "tau_ct": 10}',
        "env_overrides": '{"derate_max": 1.0}',
    }
    base_env, _, _, greedy = etf.build_track3_env(args, "./boxes/")
    assert greedy == 1.0e7
    for env in (base_env, seen["probe"]):
        assert env.derate_reference == "setpoint"
        assert env._base_dwm_params == {"tau_power": 6.0, "tau_ct": 10.0}
        assert env.derate_max == 1.0  # --env_overrides applied to the preset
    base_env.close()
    seen["probe"].close()
