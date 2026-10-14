"""Stage-14 (exp_14) presets: the two static-band twins of ``les_recipe_pin270_v2``
(static wd ~ U[band] per episode, no --train_wd_function)."""
import inspect
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import helpers.env_configs as env_configs_module  # noqa: E402
from helpers.env_configs import ENV_CONFIGS  # noqa: E402


@pytest.mark.parametrize("name, lo, hi", [
    ("les_recipe_band235_v2", 235, 270),
    ("les_recipe_band_v2", 225, 315),
])
def test_band_presets_are_pin270_v2_with_a_band(name, lo, hi):
    ref = ENV_CONFIGS["les_recipe_pin270_v2"]
    p = ENV_CONFIGS[name]
    assert p["dwm_setup"] == "les_v2"
    assert (p["wind"]["wd_min"], p["wind"]["wd_max"]) == (lo, hi)
    for k in ("ws_min", "ws_max", "TI_min", "TI_max"):
        assert p["wind"][k] == ref["wind"][k]
    assert p["farm"] == ref["farm"] and p["power_def"] == ref["power_def"]


def test_ws_max_within_8_lines_of_the_name_for_the_launcher_grep():
    src = inspect.getsource(env_configs_module).splitlines()
    for name in ("les_recipe_band235_v2", "les_recipe_band_v2"):
        i = next(j for j, l in enumerate(src) if f'"{name}":' in l)
        assert any('"ws_max"' in l for l in src[i:i + 9]), name


# ---------------------------------------------------------------------------
# Stage-14 layout-pool config fields + the trainer's include-named-layout logic
# ---------------------------------------------------------------------------

def test_stage14_layout_args_default_to_the_old_behaviour():
    from config import Args
    a = Args()
    assert a.dr_grid_spacing_lo == 4.0 and a.dr_grid_spacing_hi == 7.0
    assert a.dr_grid_exclude_D is None
    assert a.dr_include_layouts is None and a.dr_include_weight == 0.0


def _dr_args(**kw):
    from config import Args
    a = Args()
    a.layouts = "les_3x3"
    a.dr_n_lo, a.dr_n_hi, a.dr_pool_size = 4, 12, 64
    a.dr_generator = "grid"
    a.dr_grid_spacing_lo, a.dr_grid_spacing_hi = 5.0, 9.0
    a.dr_screen_headroom = False
    a.seed = 1
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def test_build_dr_layout_pool_includes_les_3x3_with_the_requested_share(capsys):
    import transformer_sac_windfarm as tsw
    from helpers.plain_turbines import make_plain_turbine
    from helpers.layouts import get_layout_positions
    wt = make_plain_turbine("IEA22")
    pool = tsw.build_dr_layout_pool(_dr_args(dr_include_layouts="les_3x3", dr_include_weight=0.125), wt)
    x0, y0 = get_layout_positions("les_3x3", wt)
    hits = [(n, x, y) for n, x, y in pool
            if x.shape == x0.shape and np.allclose(x, x0) and np.allclose(y, y0)]
    assert len(hits) == 8                                   # round(0.125 * 64)
    assert len(pool) == 64 + 8
    assert len({n for n, _, _ in pool}) == len(pool)        # names unique
    assert any(n == "les_3x3" for n, _, _ in hits)          # the first copy keeps the plain name
    assert "[layout_pool] extent from centre" in capsys.readouterr().out


def test_build_dr_layout_pool_without_include_is_the_plain_pool():
    import transformer_sac_windfarm as tsw
    from helpers.plain_turbines import make_plain_turbine
    from helpers.layout_gen import generate_layout_pool
    wt = make_plain_turbine("IEA22")
    pool = tsw.build_dr_layout_pool(_dr_args(), wt)
    ref = generate_layout_pool(64, 4, 12, float(wt.diameter()), seed=1, generator="grid",
                               grid_spacing_lo_D=5.0, grid_spacing_hi_D=9.0, screen_headroom=False)
    assert [n for n, _, _ in pool] == [n for n, _, _ in ref]
    for (_, x, y), (_, xr, yr) in zip(pool, ref):
        np.testing.assert_allclose(x, xr)


def test_build_dr_layout_pool_passes_the_exclusion_through():
    import transformer_sac_windfarm as tsw
    from helpers.plain_turbines import make_plain_turbine
    wt = make_plain_turbine("IEA22")
    D = float(wt.diameter())
    pool = tsw.build_dr_layout_pool(_dr_args(dr_n_lo=9, dr_n_hi=9, dr_pool_size=200,
                                             dr_grid_exclude_D="7:5:0.5"), wt)
    for _, x, y in pool:
        d = np.sqrt((x[:, None] - x[None, :]) ** 2 + (y[:, None] - y[None, :]) ** 2)
        d[d == 0] = np.inf
        s = np.sort(np.unique(np.round(np.sort(d, axis=1)[:, :4].ravel() / D, 3)))[:2]
        assert not (abs(s.min() - 5.0) <= 0.5 and abs(s.max() - 7.0) <= 0.5)


# ---------------------------------------------------------------------------
# Review focus 1: a band preset + an ABSOLUTE wd function must be refused by the
# trainer's pin check (transformer_sac_windfarm.py, "is ABSOLUTE"); the static
# arms carry NO --train_wd_function, the cycle arms keep the pinned preset.
# ---------------------------------------------------------------------------

def test_band_presets_with_an_absolute_function_trip_the_pin_check():
    from helpers.wd_functions import ABSOLUTE_TRAIN_NAMES, get_train_wd_factory
    import transformer_sac_windfarm as tsw
    import inspect
    src = inspect.getsource(tsw)
    assert "must pin wd_min=wd_max=" in src and "ABSOLUTE_TRAIN_NAMES" in src   # the check still exists
    f0 = float(get_train_wd_factory("cycle_270_235_slow_phase", seed=0)(0.0))
    assert "cycle_270_235_slow_phase" in ABSOLUTE_TRAIN_NAMES and f0 == 270.0
    for name in ("les_recipe_band235_v2", "les_recipe_band_v2"):
        lo, hi = (float(ENV_CONFIGS[name]["wind"][k]) for k in ("wd_min", "wd_max"))
        assert not (lo == hi == f0), name                         # the trainer's refusal condition
    lo, hi = (float(ENV_CONFIGS["les_recipe_pin270_v2"]["wind"][k]) for k in ("wd_min", "wd_max"))
    assert lo == hi == f0                                         # the cycle arms pass it
    assert "mix_static_ramp" not in ABSOLUTE_TRAIN_NAMES and "dr_ramp_narrow" not in ABSOLUTE_TRAIN_NAMES
