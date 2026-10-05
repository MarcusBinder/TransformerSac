"""Parameter-conditioning vector e (helpers/param_cond.py) and the MultiLayoutEnv
token-order parameter properties (Stage 10, RMA / UP-OSI).

Pure numpy + a fake base env; no WindFarmEnv / dynamiks needed.
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers.param_cond import (  # noqa: E402
    COND_DIM,
    COND_FARM_KEYS,
    COND_TURB_KEYS,
    CondFetcher,
    ParamNormalizer,
)
from helpers.multi_layout_env import LayoutConfig, MultiLayoutEnv  # noqa: E402

RANGES = {"yaw_exp": (-0.4, 0.6), "cp_gain": (0.95, 1.05), "tau_yaw": (0.0, 20.0)}
FARM_MEAN = {"k1": 0.09, "k2": 0.008, "d_particle": 0.65, "mann_L": 69.0, "mann_GAMMA": 3.95, "mann_AE": 0.0097}
FARM_STD = {"k1": 0.009, "k2": 0.0008, "d_particle": 0.04, "mann_L": 7.0, "mann_GAMMA": 0.18, "mann_AE": 0.0006}


def _norm(source="turbine_farm"):
    return ParamNormalizer(RANGES, FARM_MEAN, FARM_STD, clip=3.0, source=source)


def _turb_tok(max_t, rows):
    """rows: list of dicts (physical values) -> (max_t, 6) with spec defaults in the rest."""
    from WindGym.core.turbine_params import TURBINE_PARAM_SPECS
    out = np.array([[TURBINE_PARAM_SPECS[k].default for k in COND_TURB_KEYS]] * max_t, dtype=np.float64)
    for i, r in enumerate(rows):
        for k, v in r.items():
            out[i, COND_TURB_KEYS.index(k)] = v
    return out


# ----------------------------------------------------------------- constants --
def test_key_order_matches_windgym_spec_and_posterior():
    from WindGym.core.turbine_params import TURBINE_PARAM_SPECS
    assert COND_TURB_KEYS == tuple(TURBINE_PARAM_SPECS)
    assert COND_FARM_KEYS == ("k1", "k2", "d_particle", "mann_L", "mann_GAMMA", "mann_AE")
    assert COND_DIM == 12


# -------------------------------------------------------------------- vector --
def test_turbine_scaling_lo_hi_mid_and_unranged_zero():
    n = _norm()
    tok = _turb_tok(3, [{"yaw_exp": -0.4, "cp_gain": 1.05, "tau_yaw": 10.0, "ct_gain": 1.3},
                        {"yaw_exp": 0.6}, {}])
    mask = np.array([False, False, True])
    e = n.vector(tok, FARM_MEAN, mask)
    assert e.shape == (3, COND_DIM) and e.dtype == np.float32
    assert e[0, 0] == pytest.approx(-1.0)      # lo -> -1
    assert e[0, 1] == pytest.approx(+1.0)      # hi -> +1
    assert e[0, 3] == pytest.approx(0.0)       # mid -> 0
    assert e[0, 2] == 0.0                      # ct_gain has no range -> 0 even though perturbed
    assert e[1, 0] == pytest.approx(+1.0)
    assert np.all(e[2] == 0.0)                 # padded row zero (incl. farm columns)


def test_farm_zscore_clip_and_source_turbine_zeroes_farm():
    tok = _turb_tok(2, [])
    mask = np.array([False, False])
    farm = dict(FARM_MEAN)
    farm["k1"] = FARM_MEAN["k1"] + 2 * FARM_STD["k1"]
    farm["mann_L"] = FARM_MEAN["mann_L"] + 50 * FARM_STD["mann_L"]  # way out -> clipped
    e = _norm().vector(tok, farm, mask)
    assert e[0, 6] == pytest.approx(2.0)
    assert e[0, 9] == pytest.approx(3.0)
    assert e[0, 7] == pytest.approx(0.0)       # at the mean
    assert np.array_equal(e[0], e[1])          # farm params broadcast to every real token
    e_t = _norm("turbine").vector(tok, farm, mask)
    assert np.all(e_t[:, 6:] == 0.0)
    assert np.array_equal(e_t[:, :6], e[:, :6])


def test_farm_accepts_array_and_nan_maps_to_zero():
    tok = _turb_tok(1, [])
    farm_arr = np.array([FARM_MEAN[k] for k in COND_FARM_KEYS])
    farm_arr[2] = np.nan
    e = _norm().vector(tok, farm_arr, np.array([False]))
    assert np.all(np.isfinite(e)) and e[0, 8] == 0.0


def test_source_none_rejected_and_bad_shapes():
    with pytest.raises(ValueError, match="cond_source"):
        ParamNormalizer(RANGES, FARM_MEAN, FARM_STD, source="none")
    with pytest.raises(ValueError):
        _norm().vector(np.zeros((2, 5)), FARM_MEAN, np.zeros(2, bool))


def test_json_round_trip():
    n = _norm()
    s = n.to_json()
    json.loads(s)  # valid json
    m = ParamNormalizer.from_json(s)
    assert m.turb_ranges == n.turb_ranges and m.source == n.source and m.clip == n.clip
    tok = _turb_tok(2, [{"yaw_exp": 0.1, "tau_yaw": 3.0}])
    farm = {k: FARM_MEAN[k] + 0.5 * FARM_STD[k] for k in COND_FARM_KEYS}
    assert np.array_equal(n.vector(tok, farm, np.array([False, True])),
                          m.vector(tok, farm, np.array([False, True])))


def test_from_args_uses_turb_dr_ranges_and_posterior_stats():
    args = SimpleNamespace(turb_dr=("yaw_exp=-0.4:0.6", "cp_gain=0.95:1.05"), cond_source="turbine_farm")
    rng = np.random.default_rng(0)
    samples = rng.normal(size=(500, 6)) * np.array([FARM_STD[k] for k in COND_FARM_KEYS]) \
        + np.array([FARM_MEAN[k] for k in COND_FARM_KEYS])
    posterior = {"samples": samples, "names": list(COND_FARM_KEYS)}
    n = ParamNormalizer.from_args(args, posterior)
    assert n.turb_ranges == {"yaw_exp": (-0.4, 0.6), "cp_gain": (0.95, 1.05)}
    assert n.farm_mean["k1"] == pytest.approx(samples[:, 0].mean())
    assert n.farm_std["mann_L"] == pytest.approx(samples[:, 3].std())
    # turbine-only source needs no posterior
    n2 = ParamNormalizer.from_args(SimpleNamespace(turb_dr=("yaw_exp=-0.4:0.6",), cond_source="turbine"), None)
    assert n2.source == "turbine"
    # turbine_farm with a missing posterior key fails fast
    bad = {"samples": samples[:, :5], "names": list(COND_FARM_KEYS[:5])}
    with pytest.raises(ValueError, match="mann_AE"):
        ParamNormalizer.from_args(args, bad)
    with pytest.raises(ValueError, match="posterior"):
        ParamNormalizer.from_args(args, None)


# --------------------------------------------------- MultiLayoutEnv properties --
class _FakeBase(gym.Env):
    """Stands in for WindFarmEnv: per-turbine obs, turbine_params / _active_dwm_params."""

    def __init__(self, x_pos, y_pos, perturb=None, active=None, k1=0.0914):
        self.n = len(x_pos)
        self.D = 200.0
        self.observation_space = gym.spaces.Box(-1, 1, shape=(self.n, 4), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1, 1, shape=(self.n,), dtype=np.float32)
        self._perturb = perturb or {}
        self._active_dwm_params = dict(active or {})
        self.dwm_setup = SimpleNamespace(k1=k1, k2=0.0216, d_particle=0.48,
                                         mann=SimpleNamespace(L=29.4, gamma=3.9, alphaepsilon=1.0))
        self._tp = None

    @property
    def turbine_params(self):
        return self._tp

    def reset(self, *, seed=None, options=None):
        from WindGym.core.turbine_params import TURBINE_PARAM_SPECS
        self._tp = {k: np.full(self.n, TURBINE_PARAM_SPECS[k].default) for k in TURBINE_PARAM_SPECS}
        for k, v in self._perturb.items():
            self._tp[k] = np.asarray(v, dtype=np.float64)
        return np.zeros((self.n, 4), np.float32), {}

    def step(self, a):
        return np.zeros((self.n, 4), np.float32), 0.0, False, False, {}


def _mle(n=4, max_t=6, shuffle=True, seed=3, **kw):
    x = np.arange(n, dtype=np.float32) * 500
    y = np.zeros(n, np.float32)
    layout = LayoutConfig(name="row", x_pos=x, y_pos=y)
    return MultiLayoutEnv(layouts=[layout], env_factory=lambda xp, yp: _FakeBase(xp, yp, **kw),
                          per_turbine_wrapper=lambda e: e, seed=seed, shuffle=shuffle, max_turbines=max_t)


def test_turbine_params_tok_before_reset_is_defaults():
    from WindGym.core.turbine_params import TURBINE_PARAM_SPECS
    ml = _mle()
    tok = ml.turbine_params_tok
    assert tok.shape == (6, 6)
    assert np.array_equal(tok, np.array([[TURBINE_PARAM_SPECS[k].default for k in COND_TURB_KEYS]] * 6))


def test_turbine_params_tok_is_permuted_to_token_order_and_padded_with_defaults():
    phys_yaw_exp = np.array([0.1, 0.2, 0.3, 0.4])
    phys_tau = np.array([5.0, 6.0, 7.0, 8.0])
    ml = _mle(perturb={"yaw_exp": phys_yaw_exp, "tau_yaw": phys_tau})
    ml.reset(seed=3)
    perm = ml.current_permutation[: ml.n_turbines]
    assert not np.array_equal(perm, np.arange(4)) or pytest.skip("permutation happened to be identity")
    tok = ml.turbine_params_tok
    assert np.array_equal(tok[:4, 0], phys_yaw_exp[perm])
    assert np.array_equal(tok[:4, 3], phys_tau[perm])
    assert np.all(tok[4:, 0] == 0.0) and np.all(tok[4:, 1] == 1.0)   # padded rows = spec defaults
    # consistency with turbine_positions (both in token order)
    assert np.array_equal(ml.turbine_positions[:4, 0], ml.current_layout.x_pos[perm])


def test_farm_params_merge_active_over_setup_defaults():
    ml = _mle(active={"k1": 0.1, "mann_L": 50.0})
    f = ml.farm_params
    assert list(f) == list(COND_FARM_KEYS)
    assert f["k1"] == 0.1 and f["mann_L"] == 50.0          # drawn this episode
    assert f["k2"] == 0.0216 and f["d_particle"] == 0.48     # setup fallbacks
    assert f["mann_GAMMA"] == 3.9 and f["mann_AE"] == 1.0
    # a setup with k1=None (plain dynamiks) -> NaN, which the normaliser maps to 0
    ml2 = _mle(k1=None)
    assert np.isnan(ml2.farm_params["k1"])


def test_vector_from_base_env_equals_vector_of_properties():
    n = _norm()
    ml = _mle(perturb={"cp_gain": np.array([1.0, 1.05, 0.95, 1.0])}, active={"k1": 0.1})
    ml.reset(seed=3)
    e_props = n.vector(ml.turbine_params_tok, ml.farm_params, ml.attention_mask)
    e_env = n.vector_from_multilayout(ml)
    assert np.array_equal(e_props, e_env)
    perm = ml.current_permutation[:4]
    assert e_env[np.argmax(perm == 1), 1] == pytest.approx(1.0)   # physical turbine 1 (cp 1.05) -> its token


def test_cond_fetcher_refresh_stacks_per_env():
    class _Vec:
        def __init__(self, mls):
            self.mls = mls
        def get_attr(self, name):
            return [getattr(m, name) for m in self.mls]
    mls = [_mle(seed=1, active={"k1": 0.1}), _mle(seed=2, perturb={"yaw_exp": np.full(4, 0.6)})]
    for m in mls:
        m.reset()
    fetch = CondFetcher(_Vec(mls), _norm())
    assert fetch.values is None
    fetch.refresh()
    v = fetch.values
    assert v.shape == (2, 6, COND_DIM) and v.dtype == np.float32
    assert v[1, :4, 0] == pytest.approx(1.0)
    assert v[0, :4, 6] == pytest.approx((0.1 - FARM_MEAN["k1"]) / FARM_STD["k1"])
    assert np.all(v[:, 4:] == 0.0)
    out = fetch.values
    out[...] = 7.0
    assert not np.any(fetch.values == 7.0)   # refresh returns a copy, cache untouched


# ------------------------------------------------------------------- config --
def test_config_defaults_are_off():
    from config import Args
    a = Args()
    assert a.cond_source == "none" and a.cond_latent_dim == 0 and a.cond_mode == "concat"
    assert a.cond_critic == "raw" and a.cond_norm_json == "" and a.cond_action_hist is False
    assert a.cond_action_hist_len == 60 and a.adapt_rounds == 0 and a.eval_dr is False
    assert a.phase2_loss == "latent" and a.phase2_train_actor is False and a.adapt_only is False
