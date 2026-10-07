"""Every env preset selects the LES-calibrated DWM physics.

WindGym dev_dynamiks made the calibrated stack opt-in (``DWMSetup()`` is plain
dynamiks), so the presets must name it explicitly or training and eval
silently run on uncalibrated physics.
"""

import pytest

from helpers.env_configs import ENV_CONFIGS, make_env_config, require_les_calibrated

# paper-derating's older windgym pin has no calibrated_values module (the
# calibrated stack is not opt-in there); require_les_calibrated is a no-op then.
calibrated_values = pytest.importorskip("WindGym.core.calibrated_values")
LES_CALIBRATED = calibrated_values.LES_CALIBRATED
resolve_dwm_setup = calibrated_values.resolve_dwm_setup
# LES_V2 (re-calibration v2 eval point) exists on windgym >= W11 (Stage 11).
LES_V2 = getattr(calibrated_values, "LES_V2", None)
ALLOWED = tuple(s for s in (LES_CALIBRATED, LES_V2) if s is not None)


@pytest.mark.parametrize("name", sorted(ENV_CONFIGS))
def test_preset_resolves_to_a_calibrated_setup(name):
    config = make_env_config(name)
    assert resolve_dwm_setup(config.get("dwm_setup")) in ALLOWED


def test_default_presets_keep_the_keck_physics():
    # _base_config() stays les_calibrated: only the explicit v2 preset moves.
    for name in ("les_recipe", "les_recipe_pin270", "LES_curyaw_wide"):
        assert resolve_dwm_setup(make_env_config(name).get("dwm_setup")) == LES_CALIBRATED


def test_pin270_v2_resolves_to_les_v2():
    if LES_V2 is None:
        pytest.skip("windgym pin predates LES_V2")
    v2 = make_env_config("les_recipe_pin270_v2")
    base = make_env_config("les_recipe_pin270")
    assert v2["dwm_setup"] == "les_v2"
    assert resolve_dwm_setup(v2["dwm_setup"]) is LES_V2
    # everything but the setup is the pin270 recipe
    v2.pop("dwm_setup"); base.pop("dwm_setup")
    assert v2 == base


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


def test_require_les_calibrated_accepts_les_v2():
    if LES_V2 is None:
        pytest.skip("windgym pin predates LES_V2")
    require_les_calibrated("les_v2")
    require_les_calibrated(LES_V2)


@pytest.mark.parametrize("value", [None, "default"])
def test_require_les_calibrated_rejects_plain_dynamiks(value):
    with pytest.raises(RuntimeError, match="LES_CALIBRATED"):
        require_les_calibrated(value)
