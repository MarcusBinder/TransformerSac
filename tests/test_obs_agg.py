"""Tests for the LES-3x3 Stage-4 observation-aggregate wrapper (--obs_agg).

(a) pure-unit per mode on synthetic buffers (shapes, chronology, cold start,
    circular wd across the 0/360 wrap, scale ranges, EMA recursion, slopes);
(b) tiny pywake env: raw3 (L=3) == base obs, dim contract through
    PerTurbineObservationWrapper -> ObsAggWrapper -> MultiLayoutEnv, the
    truncation-step fallback, tyro parsing, mutual-exclusion errors;
(c) slow: eval_wd.create_eval_env with a synthetic checkpoint-args dict.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from helpers.obs_agg import (  # noqa: E402
    AGG_MODES, DISPERSION, EMA_TAUS, N_RAW, QUANTITIES, ObsAggWrapper,
    circmean_deg, reduce_quantity, scale_features, scale_table, wrap180,
)

TRAINED_MODES = ["latest", "mean_std", "ema", "trend", "minmax", "quantiles",
                 "raw15", "raw15span", "spectral", "spatial_rel"]
ALL_MODES = ["raw3"] + TRAINED_MODES
RANGES = {"ws": (0.0, 30.0), "wd": (0.0, 360.0), "yaw": (-45.0, 45.0),
          "power": (0.0, 10e6)}
L, DT = 30, 10.0


def _buf(rng, n=4, m=L, lo=8.0, hi=11.0):
    return rng.uniform(lo, hi, size=(m, n))


# ---------------------------------------------------------------------------
# (a) pure unit
# ---------------------------------------------------------------------------

def test_registry_matches_plan():
    assert set(AGG_MODES) == set(ALL_MODES)
    K = {m: AGG_MODES[m].K for m in AGG_MODES}
    assert K == {"raw3": 3, "latest": 1, "mean_std": 3, "ema": 4, "trend": 4,
                 "minmax": 4, "quantiles": 3, "raw15": 15, "raw15span": 15,
                 "spectral": 5, "spatial_rel": 3}


@pytest.mark.parametrize("mode", ALL_MODES)
@pytest.mark.parametrize("q", QUANTITIES)
def test_shapes_and_scaled_range(mode, q):
    rng = np.random.default_rng(0)
    lo, hi = RANGES[q]
    buf = _buf(rng, lo=lo + 0.3 * (hi - lo), hi=lo + 0.4 * (hi - lo))
    feats = reduce_quantity(mode, q, buf, L=L, dt_env=DT, front=0)
    assert feats.shape == (4, AGG_MODES[mode].K)
    assert np.isfinite(feats).all()
    scaled = scale_features(feats, AGG_MODES[mode].kinds, q, lo, hi)
    assert scaled.dtype == np.float32 and scaled.shape == feats.shape
    assert (scaled >= -1).all() and (scaled <= 1).all()


def test_scale_table_conventions():
    assert scale_table("level", "ws", 0, 30) == (0, 30)
    assert scale_table("disp", "ws", 0, 30) == (0.0, 2.0)
    assert scale_table("sdev", "wd", 0, 360) == (-30.0, 30.0)
    assert scale_table("disp", "power", 0, 8e6) == (0.0, 0.25 * 8e6)
    assert scale_table("amp", "ws", 0, 30) == (0.0, 4.0)
    # spatial_rel: yaw diffs use the level half-range, ws/wd/power fixed
    assert scale_table("rel", "yaw", -45, 45) == (-45.0, 45.0)
    assert scale_table("rel", "ws", 0, 30) == (-6.0, 6.0)
    assert scale_table("rel", "wd", 0, 360) == (-30.0, 30.0)
    assert scale_table("rel", "power", 0, 8e6) == (-0.75 * 8e6, 0.75 * 8e6)
    for q in QUANTITIES:
        assert q in DISPERSION


def test_chronology_and_latest():
    """Row -1 of the buffer is the newest sample -> the `t` feature."""
    buf = np.arange(L, dtype=float)[:, None] * np.ones((1, 3))   # 0..29 ramp
    for mode in ["latest", "mean_std", "ema", "trend", "minmax", "spectral",
                 "spatial_rel", "raw3"]:
        f = reduce_quantity(mode, "ws", buf, L=L, dt_env=DT, front=0)
        assert np.allclose(f[:, 0], 29.0), mode
    r3 = reduce_quantity("raw3", "ws", buf, L=L, dt_env=DT, front=0)
    assert np.allclose(r3[0], [29, 28, 27])
    r15 = reduce_quantity("raw15", "ws", buf, L=L, dt_env=DT, front=0)
    assert np.allclose(r15[0], np.arange(15, 30))                # chronological
    q = reduce_quantity("quantiles", "ws", buf, L=L, dt_env=DT, front=0)
    assert np.allclose(q[0], np.quantile(np.arange(30.0), [0.1, 0.5, 0.9]))
    mm = reduce_quantity("minmax", "ws", buf, L=L, dt_env=DT, front=0)
    assert np.allclose(mm[0], [29, 0, 29, 14.5])


def test_trend_slope_on_linear_ramp():
    buf = 2.0 * np.arange(L, dtype=float)[:, None] * np.ones((1, 2)) + 5.0
    f = reduce_quantity("trend", "ws", buf, L=L, dt_env=DT, front=0)
    # [t, mean_6, slope_6*6, slope_30*30]: slope 2/step -> 12 and 60
    assert np.allclose(f[0], [63.0, 2 * 26.5 + 5, 12.0, 60.0])


def test_ema_recursion():
    rng = np.random.default_rng(1)
    buf = _buf(rng, n=1)
    f = reduce_quantity("ema", "ws", buf, L=L, dt_env=DT, front=0)
    for j, tau in enumerate(EMA_TAUS):
        a = 1 - np.exp(-DT / tau)
        e = buf[0, 0]
        for x in buf[1:, 0]:
            e = (1 - a) * e + a * x
        assert np.isclose(f[0, 1 + j], e)
    # a constant buffer has EMA == the constant at every scale
    c = reduce_quantity("ema", "ws", np.full((L, 2), 9.5), L=L, dt_env=DT, front=0)
    assert np.allclose(c, 9.5)


def test_mean_std_basic():
    rng = np.random.default_rng(2)
    buf = _buf(rng)
    f = reduce_quantity("mean_std", "ws", buf, L=L, dt_env=DT, front=0)
    assert np.allclose(f[:, 1], buf.mean(axis=0))
    assert np.allclose(f[:, 2], buf.std(axis=0))          # ddof=0
    one = reduce_quantity("mean_std", "ws", buf[-1:], L=L, dt_env=DT, front=0)
    assert np.allclose(one[:, 2], 0.0)                     # m == 1 -> std 0


def test_spectral_amplitude_of_a_sinusoid():
    t = np.arange(L)
    buf = (9.0 + 0.5 * np.sin(2 * np.pi * 2 * t / L))[:, None]   # period L/2
    f = reduce_quantity("spectral", "ws", buf, L=L, dt_env=DT, front=0)
    assert np.isclose(f[0, 1], 9.0)
    assert np.isclose(f[0, 3], 0.5, atol=1e-6)             # |F_2| ~ amplitude
    assert f[0, 2] < 1e-6 and f[0, 4] < 1e-6


def test_spatial_rel_uses_front_and_farm_mean():
    buf = np.tile(np.array([[10.0, 8.0, 6.0]]), (L, 1))
    f = reduce_quantity("spatial_rel", "ws", buf, L=L, dt_env=DT, front=0)
    assert np.allclose(f[:, 0], [10, 8, 6])
    assert np.allclose(f[:, 1], [0, -2, -4])
    assert np.allclose(f[:, 2], [2, 0, -2])
    # wd: differences wrapped
    wbuf = np.tile(np.array([[358.0, 2.0, 10.0]]), (L, 1))
    g = reduce_quantity("spatial_rel", "wd", wbuf, L=L, dt_env=DT, front=0)
    assert np.allclose(g[:, 1], [0, 4, 12])
    cm = circmean_deg(np.array([358.0, 2.0, 10.0]))
    assert np.allclose(g[:, 2], wrap180(np.array([358.0, 2.0, 10.0]) - cm))


def test_circular_wd_across_the_wrap():
    rng = np.random.default_rng(3)
    # buffer straddling 0/360: 358..2 deg
    buf = (rng.uniform(-2.0, 2.0, size=(L, 3))) % 360.0
    f = reduce_quantity("mean_std", "wd", buf, L=L, dt_env=DT, front=0)
    mean = f[:, 1]
    assert np.all((mean < 3.0) | (mean > 357.0)), mean      # ~0, not ~180
    assert np.all(f[:, 2] < 2.5)                            # small dispersion
    mm = reduce_quantity("minmax", "wd", buf, L=L, dt_env=DT, front=0)
    assert np.all((mm[:, 1] > 355.0)) and np.all(mm[:, 2] < 5.0)   # min ~358, max ~2
    tr = reduce_quantity("trend", "wd", buf, L=L, dt_env=DT, front=0)
    assert np.all(np.abs(tr[:, 2:]) < 30.0)
    # a wd ramp crossing 360 keeps a clean slope
    ramp = (np.linspace(350.0, 370.0, L) % 360.0)[:, None]
    tr = reduce_quantity("trend", "wd", ramp, L=L, dt_env=DT, front=0)
    assert np.isclose(tr[0, 3], 20.0 / (L - 1) * L, atol=1e-6)   # slope*L
    ema = reduce_quantity("ema", "wd", ramp, L=L, dt_env=DT, front=0)
    assert np.all((ema[0] > 350.0) | (ema[0] < 10.5))
    # levels are in [0, 360) and env-scale like today
    for m in ["latest", "quantiles", "raw15", "raw15span", "spectral"]:
        g = reduce_quantity(m, "wd", buf, L=L, dt_env=DT, front=0)
        lv = [k for k, kind in enumerate(AGG_MODES[m].kinds) if kind == "level"]
        assert np.all(g[:, lv] >= 0) and np.all(g[:, lv] < 360)


def test_short_buffers():
    rng = np.random.default_rng(4)
    for m in (1, 2, 5):
        buf = _buf(rng, m=m)
        for mode in ALL_MODES:
            f = reduce_quantity(mode, "ws", buf, L=L, dt_env=DT, front=0)
            assert f.shape == (4, AGG_MODES[mode].K) and np.isfinite(f).all()
    one = _buf(rng, m=1)
    r15 = reduce_quantity("raw15", "ws", one, L=L, dt_env=DT, front=0)
    assert np.allclose(r15, one[0][:, None])               # oldest repeated
    r15s = reduce_quantity("raw15span", "ws", one, L=L, dt_env=DT, front=0)
    assert np.allclose(r15s, one[0][:, None])              # oldest repeated
    tr = reduce_quantity("trend", "ws", one, L=L, dt_env=DT, front=0)
    assert np.allclose(tr[:, 2:], 0.0)                     # m<2 -> slope 0
    assert reduce_quantity("spectral", "ws", one, L=L, dt_env=DT, front=0).shape == (4, 5)


# ---------------------------------------------------------------------------
# (a') raw15span (Stage 6: 15 samples spanning the whole L-buffer)
# ---------------------------------------------------------------------------

def test_raw15span_takes_linspace_indices_over_the_buffer():
    Lspan = 60
    buf = np.arange(Lspan, dtype=float)[:, None] * np.ones((1, 3))  # 0..59 ramp
    f = reduce_quantity("raw15span", "ws", buf, L=Lspan, dt_env=DT, front=0)
    idx = np.round(np.linspace(0, Lspan - 1, N_RAW)).astype(int)
    assert np.allclose(f[0], idx.astype(float))            # chronological
    assert f[0, -1] == 59.0                                # newest included
    assert f[0, 0] == 0.0                                  # oldest included
    # with obs_agg_len 60 @ dt_env 10 the window spans 600 s at ~40-s spacing
    assert np.all(np.diff(idx) >= 4)


def test_raw15span_equals_raw15_at_L15():
    """The design invariant that lets raw15 checkpoints stay bit-reproducible:
    at L == 15 the linspace indices are 0..14, so the two modes coincide."""
    rng = np.random.default_rng(6)
    for q in QUANTITIES:
        lo, hi = RANGES[q]
        for m in (1, 3, 15, 20):
            buf = _buf(rng, m=m, lo=lo + 0.3 * (hi - lo), hi=lo + 0.4 * (hi - lo))
            a = reduce_quantity("raw15", q, buf, L=15, dt_env=DT, front=0)
            b = reduce_quantity("raw15span", q, buf, L=15, dt_env=DT, front=0)
            np.testing.assert_array_equal(a, b)


def test_raw15span_short_buffer_pads_with_the_oldest_sample():
    Lspan = 60
    m = 10
    buf = np.arange(m, dtype=float)[:, None] * np.ones((1, 2)) + 3.0  # 3..12
    f = reduce_quantity("raw15span", "ws", buf, L=Lspan, dt_env=DT, front=0)
    # padded buffer = [3]*50 + [3..12]; linspace indices < 50 all hit the pad
    idx = np.round(np.linspace(0, Lspan - 1, N_RAW)).astype(int)
    padded = np.concatenate([np.full(Lspan - m, 3.0), np.arange(m) + 3.0])
    assert np.allclose(f[0], padded[idx])
    assert f[0, -1] == 12.0                                # newest still last


# ---------------------------------------------------------------------------
# (b) tiny pywake env
# ---------------------------------------------------------------------------

X_POS = [0.0, 8 * 178.3, 16 * 178.3]
Y_POS = [0.0, 0.0, 0.0]


@pytest.fixture(scope="module")
def wind_turbine():
    from py_wake.examples.data.dtu10mw import DTU10MW
    return DTU10MW()


def _config(history_length=3, deque_len=None):
    from helpers.env_configs import make_env_config
    config = make_env_config("les_recipe")
    config["wind"]["ws_min"] = config["wind"]["ws_max"] = 9.5
    config["wind"]["wd_min"] = config["wind"]["wd_max"] = 270.0
    for p in ("ws", "wd", "yaw", "power"):
        config[f"{p}_mes"][f"{p}_history_N"] = history_length
        config[f"{p}_mes"][f"{p}_history_length"] = deque_len or history_length
    return config


def _base_env(wind_turbine, config, n_passthrough=3):
    from WindGym import WindFarmEnv
    return WindFarmEnv(x_pos=X_POS, y_pos=Y_POS, turbine=wind_turbine,
                       config=config, backend="pywake", turbtype="Random",
                       TurbBox="Default", dt_sim=10, dt_env=10, yaw_step_sim=5,
                       n_passthrough=n_passthrough, reset_init=False)


def test_raw3_equals_base_obs(wind_turbine):
    from WindGym.wrappers import PerTurbineObservationWrapper
    env = _base_env(wind_turbine, _config(3, 3))
    per = PerTurbineObservationWrapper(env)
    agg = ObsAggWrapper(per, "raw3", 3, dt_env=10.0)
    try:
        assert agg._obs_dim_per_turbine == 12
        obs, _ = agg.reset(seed=0)
        base_obs = per._reshape_obs_to_per_turbine(env._get_obs())
        np.testing.assert_allclose(obs, base_obs, atol=1e-6)
        rng = np.random.default_rng(0)
        for _ in range(20):
            a = rng.uniform(-1, 1, size=agg.action_space.shape).astype(np.float32)
            obs, r, term, trunc, info = agg.step(a)
            base_obs = per._reshape_obs_to_per_turbine(env._get_obs())
            np.testing.assert_allclose(obs, base_obs, atol=1e-6)
            assert not trunc
    finally:
        env.close()


def test_dim_contract_through_multilayout(wind_turbine):
    from WindGym.wrappers import PerTurbineObservationWrapper
    from helpers.multi_layout_env import LayoutConfig, MultiLayoutEnv

    config = _config(3, 30)

    def env_factory(x, y):
        from WindGym import WindFarmEnv
        return WindFarmEnv(x_pos=x, y_pos=y, turbine=wind_turbine, config=config,
                           backend="pywake", turbtype="Random", TurbBox="Default",
                           dt_sim=10, dt_env=10, yaw_step_sim=5, n_passthrough=3,
                           reset_init=False)

    def wrapper(env):
        env = PerTurbineObservationWrapper(env)
        return ObsAggWrapper(env, "mean_std", 30, dt_env=10.0)

    layout = LayoutConfig(name="row3", x_pos=X_POS, y_pos=Y_POS)
    ml = MultiLayoutEnv(layouts=[layout], env_factory=env_factory,
                        per_turbine_wrapper=wrapper, seed=0, shuffle=True,
                        max_turbines=5, max_episode_steps=50)
    try:
        assert ml.obs_dim_per_turbine == 12
        obs, _ = ml.reset(seed=0)
        assert obs.shape == (5, 12)
        assert np.all(obs[3:] == 0.0)                      # pad rows zero
        # a full 30-deep buffer right after reset (burn-in filled it)
        inner = ml._current_env
        m = len(inner.env.unwrapped.farm_measurements.turb_mes[0].ws.measurements)
        assert m == 30
        assert np.all(np.abs(obs[:3]) <= 1.0)
        # std of ws over a static-inflow pywake env is small but the mean and
        # latest columns carry the 9.5 m/s level
        ws_level = (obs[:3, 0] + 1) / 2 * 30
        assert np.all(np.abs(ws_level - 9.5) < 3.0)
        for _ in range(3):
            obs, *_ = ml.step(np.zeros(ml.action_space.shape, dtype=np.float32))
            assert obs.shape == (5, 12) and np.all(np.abs(obs) <= 1.0)
    finally:
        ml.close()


def test_truncation_step_falls_back_to_last_obs(wind_turbine):
    """The 1-passthrough pywake env truncates after ~14 steps and frees
    farm_measurements; the wrapper must still return a valid obs."""
    from WindGym.wrappers import PerTurbineObservationWrapper
    env = _base_env(wind_turbine, _config(3, 30), n_passthrough=1)
    agg = ObsAggWrapper(PerTurbineObservationWrapper(env), "latest", 30, dt_env=10.0)
    try:
        agg.reset(seed=0)
        trunc, last, n = False, None, 0
        while not trunc and n < 200:
            obs, r, term, trunc, info = agg.step(
                np.zeros(agg.action_space.shape, dtype=np.float32))
            n += 1
            if not trunc:
                last = obs
        assert trunc and env.farm_measurements is None
        np.testing.assert_array_equal(obs, last)
    finally:
        env.close()


def test_rejects_short_deques(wind_turbine):
    from WindGym.wrappers import PerTurbineObservationWrapper
    env = _base_env(wind_turbine, _config(3, 3))
    try:
        with pytest.raises(ValueError, match="obs_agg_len"):
            ObsAggWrapper(PerTurbineObservationWrapper(env), "mean_std", 30, dt_env=10.0)
        with pytest.raises(ValueError, match="Unknown obs_agg"):
            ObsAggWrapper(PerTurbineObservationWrapper(env), "nope", 3, dt_env=10.0)
    finally:
        env.close()


def test_tyro_flags_and_defaults():
    import tyro
    from config import Args
    a = tyro.cli(Args, args=[])
    assert a.obs_agg is None and a.obs_agg_len == 30
    b = tyro.cli(Args, args=["--obs_agg", "mean_std", "--obs_agg_len", "60"])
    assert b.obs_agg == "mean_std" and b.obs_agg_len == 60


def test_mutual_exclusion_source_guards():
    """The trainer hard-errors on the forbidden combos; guard the text so the
    checks cannot silently be dropped."""
    src = open(os.path.join(os.path.dirname(__file__), "..",
                            "transformer_sac_windfarm.py")).read()
    for needle in ("--obs_agg cannot be combined with --obs_encoding",
                   "--obs_agg cannot be combined with --use_wd_deviation",
                   "per_sensor asserts obs_dim",
                   "--obs_agg cannot be combined with the DEL wrapper"):
        assert needle in src, needle


# ---------------------------------------------------------------------------
# (c) eval_wd round trip
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.parametrize("obs_agg,expect", [(None, 12), ("mean_std", 12),
                                            ("raw15", 60)])
def test_create_eval_env_rebuilds_the_wrapper(obs_agg, expect):
    import types
    import gymnasium as gym
    from config import Args
    import eval_wd
    from helpers.wd_functions import get_wd_function

    args = vars(Args())
    args.update({"config": "les_recipe", "turbtype": "DTU10MW", "history_length": 3,
                 "backend": "pywake", "dt_sim": 10, "dt_env": 10, "yaw_step": 5,
                 "max_eps": 3, "max_turbines": 4, "profile_encoding_type": None,
                 "obs_agg": obs_agg, "obs_agg_len": 30})
    cli = types.SimpleNamespace(ws=9.5, backend=None, turbbox_path="Default",
                                TI_type=None, dt_sim=None, dt_env=None,
                                wd_source="true", wd_est_tau=None,
                                wd_est_consensus=None, max_turb_move=None)
    env, _ = eval_wd.create_eval_env("test_layout", args, cli,
                                     get_wd_function("static_270"), seed=0,
                                     n_envs=1, vector_cls=gym.vector.SyncVectorEnv)
    try:
        assert env.single_observation_space.shape[-1] == expect
        obs, _ = env.reset(seed=0)
        assert obs.shape[-1] == expect and np.all(np.abs(obs) <= 1.0)
    finally:
        env.close()
