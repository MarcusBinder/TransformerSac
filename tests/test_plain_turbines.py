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
