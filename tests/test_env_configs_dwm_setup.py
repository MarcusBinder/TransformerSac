"""Every env preset selects the LES-calibrated DWM physics.

WindGym dev_dynamiks made the calibrated stack opt-in (``DWMSetup()`` is plain
dynamiks), so the presets must name it explicitly or training and eval
silently run on uncalibrated physics.
"""

import pytest

from helpers.env_configs import ENV_CONFIGS, make_env_config, require_les_calibrated

from WindGym.core.calibrated_values import LES_CALIBRATED, resolve_dwm_setup


@pytest.mark.parametrize("name", sorted(ENV_CONFIGS))
def test_preset_resolves_to_les_calibrated(name):
    config = make_env_config(name)
    assert resolve_dwm_setup(config.get("dwm_setup")) == LES_CALIBRATED


def test_env_built_from_preset_uses_les_calibrated():
    import numpy as np
    from py_wake.examples.data.hornsrev1 import V80

    from WindGym import WindFarmEnv

    d = V80().diameter()
    env = WindFarmEnv(
        turbine=V80(),
        x_pos=np.array([0.0, 5 * d]),
        y_pos=np.zeros(2),
        config=make_env_config("LES_curyaw_wide"),
        turbtype="None",
        reset_init=False,
    )
    assert env.dwm_setup == LES_CALIBRATED


@pytest.mark.parametrize("value", ["les_calibrated", LES_CALIBRATED])
def test_require_les_calibrated_accepts(value):
    require_les_calibrated(value)


@pytest.mark.parametrize("value", [None, "default"])
def test_require_les_calibrated_rejects_plain_dynamiks(value):
    with pytest.raises(RuntimeError, match="LES_CALIBRATED"):
        require_les_calibrated(value)
