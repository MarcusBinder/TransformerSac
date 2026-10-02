"""Turbine-parameter DR sampler + CLI spec parsing (helpers/dr_turbine.py)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers.dr_posterior import make_dr_sampler  # noqa: E402
from helpers.dr_turbine import (  # noqa: E402
    combine_samplers,
    format_turb_dr,
    make_turbine_sampler,
    parse_turb_dr,
    validate_turb_dr_names,
)

SPEC = ("yaw_exp=-0.4:0.6", "cp_gain=0.95:1.05", "tau_yaw=0:20")


# ------------------------------------------------------------------ parsing --
def test_parse_round_trip_and_empty():
    r = parse_turb_dr(SPEC)
    assert r == {"yaw_exp": (-0.4, 0.6), "cp_gain": (0.95, 1.05), "tau_yaw": (0.0, 20.0)}
    assert list(r) == ["yaw_exp", "cp_gain", "tau_yaw"]  # order kept
    assert parse_turb_dr(()) == {} and parse_turb_dr(None) == {}


@pytest.mark.parametrize("bad, match", [
    (("yaw_exp=0.5",), "name=lo:hi"),
    (("yaw_exp:0:1",), "name=lo:hi"),
    (("yaw_exp=a:b",), "yaw_exp"),
    (("yaw_exp=0.6:-0.4",), "lo > hi"),
    (("yaw_exp=0:1", "yaw_exp=0:2"), "duplicate"),
])
def test_parse_errors(bad, match):
    with pytest.raises(ValueError, match=match):
        parse_turb_dr(bad)


def test_format_banner():
    s = format_turb_dr(parse_turb_dr(SPEC))
    assert s.startswith("[TurbDR]") and "yaw_exp~U[-0.4,0.6]" in s and "tau_yaw~U[0,20]" in s


# --------------------------------------------------------------- validation --
class _FakeSpec:
    def __init__(self, lo, hi):
        self.lo, self.hi = lo, hi


FAKE_SPECS = {"yaw_exp": _FakeSpec(-3, 3), "cp_gain": _FakeSpec(0.2, 2.0)}


def test_validate_unknown_name_against_injected_spec():
    with pytest.raises(ValueError, match="Unknown turbine DR param.*tau_yaw"):
        validate_turb_dr_names({"tau_yaw": (0, 1)}, spec=FAKE_SPECS)


def test_validate_range_outside_spec_bounds():
    with pytest.raises(ValueError, match=r"cp_gain.*\(0.1, 1.0\).*\[0.2, 2.0\]"):
        validate_turb_dr_names({"cp_gain": (0.1, 1.0)}, spec=FAKE_SPECS)
    validate_turb_dr_names({"cp_gain": (0.2, 2.0), "yaw_exp": (-3, 3)}, spec=FAKE_SPECS)  # edges ok


def test_validate_against_real_windgym_spec():
    pytest.importorskip("WindGym.core.turbine_params")
    validate_turb_dr_names(parse_turb_dr(SPEC))
    with pytest.raises(ValueError, match="Unknown turbine DR param"):
        validate_turb_dr_names({"bogus": (0, 1)})
    validate_turb_dr_names({})  # empty never touches windgym


def test_validate_missing_module_with_ranges_is_version_skew(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name.startswith("WindGym"):
            raise ImportError("no windgym")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(RuntimeError, match="turbine_params"):
        validate_turb_dr_names({"cp_gain": (0.9, 1.1)})
    validate_turb_dr_names({})  # no ranges -> no import -> fine


# ----------------------------------------------------------------- sampler --
def test_sampler_shapes_bounds_and_determinism():
    ranges = parse_turb_dr(SPEC)
    s = make_turbine_sampler(ranges, n_turb=9)
    d1 = s(np.random.default_rng(5))
    d2 = s(np.random.default_rng(5))
    assert set(d1) == set(ranges)
    for k, (lo, hi) in ranges.items():
        assert d1[k].shape == (9,) and d1[k].dtype == np.float64
        assert np.all(d1[k] >= lo) and np.all(d1[k] <= hi)
        np.testing.assert_array_equal(d1[k], d2[k])
    assert len(np.unique(d1["yaw_exp"])) == 9  # iid per turbine


def test_sampler_farm_wide_floats():
    s = make_turbine_sampler(parse_turb_dr(SPEC), n_turb=9, per_turbine=False)
    d = s(np.random.default_rng(0))
    assert all(isinstance(v, float) for v in d.values())
    assert -0.4 <= d["yaw_exp"] <= 0.6


# ---------------------------------------------------------------- combine --
def _posterior():
    rng = np.random.default_rng(1)
    return {"samples": rng.normal(size=(50, 3)), "names": ["k1", "k2", "d_particle"]}


def test_combine_none_and_posterior_only_bit_identical():
    assert combine_samplers(None, None) is None
    post = make_dr_sampler(_posterior(), keys=("k1", "k2"))
    comb = combine_samplers(post, None)
    for seed in (0, 1, 2):
        assert comb(np.random.default_rng(seed)) == post(np.random.default_rng(seed))


def test_combine_merges_posterior_then_turbine_on_one_rng():
    post = make_dr_sampler(_posterior(), keys=("k1", "k2"))
    turb = make_turbine_sampler(parse_turb_dr(SPEC), n_turb=3)
    comb = combine_samplers(post, turb)
    d = comb(np.random.default_rng(3))
    assert set(d) == {"k1", "k2", "yaw_exp", "cp_gain", "tau_yaw"}
    # posterior draws first on the shared rng: identical to the standalone draw
    rng = np.random.default_rng(3)
    assert {k: d[k] for k in ("k1", "k2")} == post(rng)
    expect_turb = turb(rng)
    for k in expect_turb:
        np.testing.assert_array_equal(d[k], expect_turb[k])


def test_combine_overlapping_keys_raise():
    a = lambda rng: {"k1": 1.0}  # noqa: E731
    b = lambda rng: {"k1": 2.0, "cp_gain": np.ones(2)}  # noqa: E731
    with pytest.raises(KeyError, match="k1"):
        combine_samplers(a, b)(np.random.default_rng(0))


def test_wrapper_smoke_on_stub_env():
    gym = pytest.importorskip("gymnasium")
    from WindGym.wrappers import DWMRandomizationWrapper

    class Stub(gym.Env):
        observation_space = gym.spaces.Box(-1, 1, (1,), np.float32)
        action_space = gym.spaces.Box(-1, 1, (1,), np.float32)

        def reset(self, *, seed=None, options=None):
            self.last_options = options
            return np.zeros(1, np.float32), {}

        def step(self, a):
            return np.zeros(1, np.float32), 0.0, False, False, {}

    stub = Stub()
    sampler = combine_samplers(make_dr_sampler(_posterior(), keys=("k1", "k2", "d_particle")),
                               make_turbine_sampler(parse_turb_dr(SPEC), n_turb=3))
    w = DWMRandomizationWrapper(stub, sampler=sampler, seed=11)
    w.reset()
    sent = stub.last_options["dwm_params"]
    assert set(sent) == {"k1", "k2", "d_particle", "yaw_exp", "cp_gain", "tau_yaw"}
    assert sent["cp_gain"].shape == (3,) and isinstance(sent["k1"], float)
    np.testing.assert_array_equal(w.last_theta["cp_gain"], sent["cp_gain"])
