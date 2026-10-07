"""Stage-11 DR contract: the re-calibration v2 posterior (RECAL_V2 3.8) feeding
the 7-key randomiser and the 13-dim conditioning vector.

Skips when the posterior is not staged (it is a gitignored laptop artifact in
LESRL and a tracked root file in paper-derating).
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers.dr_posterior import load_posterior, make_dr_sampler  # noqa: E402
from helpers.param_cond import COND_FARM_KEYS, ParamNormalizer, farm_params_from_base_env  # noqa: E402

_ROOT = Path(__file__).resolve().parents[2]
CANDIDATES = (_ROOT / "posterior_v2_samples.npz", _ROOT / "results" / "calibration_v2" / "posterior_samples.npz")
V2_KEYS = ("k1", "k2", "d_particle", "hill_vortex_factor", "mann_L", "mann_GAMMA", "mann_AE")
CONST = ("k1", "d_particle")


@pytest.fixture(scope="module")
def posterior():
    for p in CANDIDATES:
        if p.exists():
            return load_posterior(str(p))
    pytest.skip(f"v2 posterior not staged at any of {[str(c) for c in CANDIDATES]}")


def test_v2_contract(posterior):
    s = posterior["samples"]
    assert s.shape == (2000, 7)
    assert sorted(posterior["names"]) == sorted(V2_KEYS)
    assert np.all(np.isfinite(s))
    names = posterior["names"]
    for k in CONST:
        assert s[:, names.index(k)].std() < 1e-9, k
    c = s[:, names.index("hill_vortex_factor")]
    assert c.min() >= 0.27 and c.max() <= 0.33
    assert s[:, names.index("k2")].mean() == pytest.approx(0.0202, abs=5e-4)
    assert s[:, names.index("mann_L")].mean() == pytest.approx(72.7, abs=0.5)


def test_seven_key_sampler_draws_a_joint_row(posterior):
    sampler = make_dr_sampler(posterior, keys=V2_KEYS)
    draw = sampler(np.random.default_rng(11))
    assert set(draw) == set(V2_KEYS)
    cols = [posterior["names"].index(k) for k in V2_KEYS]
    row = np.array([draw[k] for k in V2_KEYS])
    assert np.all(np.isclose(posterior["samples"][:, cols], row[None, :]), axis=1).any()


def test_validate_dr_setup_passes_silently(posterior, capsys):
    import transformer_sac_windfarm as tsw
    args = SimpleNamespace(backend="dynamiks", dr_keys=V2_KEYS, TI_type="MannGenerate")
    tsw.validate_dr_setup(args, posterior)
    assert "[DR] WARNING" not in capsys.readouterr().out


def test_expert_gate_les_v2_inside_the_posterior(posterior):
    """The eval point LES_V2 z-scored by the v2 posterior: |z| < 3 everywhere
    (the policy's expert / nominal e is inside the training support, unlike the
    Stage-10 Keck point) and exactly 0 on the constant columns."""
    cv = pytest.importorskip("WindGym.core.calibrated_values")
    if not hasattr(cv, "LES_V2"):
        pytest.skip("windgym pin predates LES_V2")
    args = SimpleNamespace(turb_dr=("yaw_exp=-0.4:0.6",), cond_source="turbine_farm", cond_clip=3.0)
    n = ParamNormalizer.from_args(args, posterior)
    assert n.farm_keys == COND_FARM_KEYS and n.cond_dim == 13
    farm = farm_params_from_base_env(SimpleNamespace(dwm_setup=cv.LES_V2, active_dwm_params=None))
    assert set(farm) == set(V2_KEYS)
    e = n.vector(np.zeros((1, 6)), farm, np.array([False]))
    z = e[0, 6:]
    assert np.all(np.abs(z) < 3.0), z
    for k in CONST:
        assert z[COND_FARM_KEYS.index(k)] == 0.0, k
