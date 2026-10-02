"""Tests for helpers/plain_turbines.make_plain_turbine (Stage 6: IEA22 port).

The IEA22 numbers are parity anchors against the LESRL vendor files
(helpers/iea_22_rwt.py + iea_22_rwt.pwr, copied verbatim): D=284 m, hub
170 m, ~22 MW electrical rating (the .pwr aero power x 0.95429... factor).
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from helpers.plain_turbines import make_plain_turbine  # noqa: E402


def test_iea22_geometry_and_power():
    wt = make_plain_turbine("IEA22")
    assert float(wt.diameter()) == pytest.approx(284.0)
    assert float(wt.hub_height()) == pytest.approx(170.0)
    # Electrical rating ~22 MW (0.9542919819763047 x aero .pwr table).
    ws = np.arange(3.0, 25.5, 0.5)
    pmax = float(np.max(wt.power(ws)))
    assert pmax == pytest.approx(22e6, rel=0.05)
    # Below-rated operating point of the campaign envelope (ws 8-11, eval 9).
    p9 = float(wt.power(9.0))
    assert 5e6 < p9 < pmax
    ct9 = float(wt.ct(9.0))
    assert 0.0 < ct9 < 1.2


def test_iea22_unchanged():
    # Stage-8 anchor: the 1-D H2S curve must not move when the surrogate lands
    # (P(9.2, yaw=0) from the paper-derating .venv, py_wake 2.6.20).
    wt = make_plain_turbine("IEA22")
    assert float(np.ravel(wt.power(9.2))[0]) == pytest.approx(14.155e6, rel=0.01)


def test_iea22h2_surrogate():
    # Lowercase is what the Stage-8 launcher passes (LESRL harnesses compare
    # turbtype.lower()); the upper alias must map to the same class.
    wt = make_plain_turbine("iea22h2")
    assert type(make_plain_turbine("IEA22H2")) is type(wt)
    assert str(wt.name()) == "IEA_22MW_280_RWT_HAWC2S"
    assert float(wt.diameter()) == pytest.approx(284.0)
    assert float(wt.hub_height()) == pytest.approx(170.0)
    # HAWC2 aero power at the precursor hub speed, zero yaw.
    p0 = float(np.ravel(wt.power(9.2, yaw=0.0))[0])
    assert p0 == pytest.approx(13.663e6, rel=0.01)
    # The table carries the yaw loss (cos^1.65-ish, NOT PyWake's cos^2.9).
    p20 = float(np.ravel(wt.power(9.2, yaw=20.0))[0])
    assert p20 / p0 == pytest.approx(0.908, abs=0.01)
    assert float(np.ravel(wt.ct(9.2, yaw=0.0))[0]) == pytest.approx(0.7525, rel=0.01)
    # yaw is an OPTIONAL input with default 0 -> dynamiks forwards the sensor.
    pcm = wt.powerCtFunction
    assert pcm.default_value_dict == {"yaw": 0.0}
    assert "yaw" in pcm.optional_inputs
    # Aero power_max 23.1 MW (windgym maxturbpower; IEA22 electrical is 22.0).
    pmax = float(np.max(wt.power(np.arange(10.0, 25.0), yaw=0.0)))
    assert pmax == pytest.approx(23.1e6, rel=0.01)


def test_dtu10mw_unchanged():
    wt = make_plain_turbine("DTU10MW")
    assert float(wt.diameter()) == pytest.approx(178.3)
    assert float(np.max(wt.power(np.arange(4.0, 25.0)))) == pytest.approx(
        10e6, rel=0.05)


def test_v80_dispatch():
    wt = make_plain_turbine("V80")
    assert float(wt.diameter()) == pytest.approx(80.0)


def test_unknown_turbtype_raises():
    with pytest.raises(ValueError, match="IEA34"):
        # IEA34 is the DERATE-capable default; the plain dispatch must reject
        # it loudly (derate configs go through make_derating_turbine instead).
        make_plain_turbine("IEA34")
    with pytest.raises(ValueError, match="nope"):
        make_plain_turbine("nope")
