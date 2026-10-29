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
    # checkpoints saved before the flags existed back-fill to None / "{}"
    assert turbine_model_spec({}) == {"derate_reference": None, "turbine_params": {}}
