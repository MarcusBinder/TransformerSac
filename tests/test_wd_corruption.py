"""Unit tests for helpers/wd_corruption.py (T0 sensitivity injection).

Checks the three contracts eval_wd.py relies on:
  * bias is exact,
  * AR(1) noise is stationary with the requested sigma (tau-independent),
  * lag is an exact shift on a ramp, pre-filled at reset.
"""
import numpy as np
import pytest

from helpers.wd_corruption import (
    Ar1Corruptor,
    BiasCorruptor,
    LagCorruptor,
    WdCorruptor,
    make_corruptor,
)


N_ENVS = 4
DT = 10.0


def test_none_is_identity():
    c = make_corruptor("none", N_ENVS, DT)
    wd = np.array([270.0, 280.0, 290.0, 300.0])
    c.reset(wd)
    out = c.corrupt(wd)
    np.testing.assert_array_equal(out, wd)
    # returns a copy, never an alias of the live env array
    out[:] = 0.0
    assert wd[0] == 270.0


def test_bias_exact():
    for bias in (2.0, -5.0, 10.0):
        c = make_corruptor(f"bias:{bias}", N_ENVS, DT)
        wd = np.full(N_ENVS, 270.0)
        c.reset(wd)
        for _ in range(5):
            np.testing.assert_allclose(c.corrupt(wd), wd + bias)


def test_ar1_stationary_sigma():
    """Long-run std matches the spec sigma regardless of tau."""
    sigma = 5.0
    for tau in (30.0, 60.0, 300.0):
        c = Ar1Corruptor(n_envs=N_ENVS, dt_env=DT, sigma_deg=sigma,
                         tau_s=tau, seed=123)
        wd = np.full(N_ENVS, 270.0)
        c.reset(wd)
        errs = np.concatenate(
            [c.corrupt(wd) - wd for _ in range(20000)])
        assert abs(errs.std() - sigma) < 0.15 * sigma, (
            f"tau={tau}: stationary std {errs.std():.2f} != sigma {sigma}")
        assert abs(errs.mean()) < 0.05 * sigma * 3  # zero-mean


def test_ar1_autocorrelation_matches_tau():
    """One-step autocorrelation is exp(-dt/tau)."""
    tau, sigma = 60.0, 5.0
    c = Ar1Corruptor(n_envs=1, dt_env=DT, sigma_deg=sigma, tau_s=tau, seed=7)
    wd = np.zeros(1)
    c.reset(wd)
    e = np.array([c.corrupt(wd)[0] for _ in range(200000)])
    rho_hat = np.corrcoef(e[:-1], e[1:])[0, 1]
    assert abs(rho_hat - np.exp(-DT / tau)) < 0.01


def test_ar1_reset_redraws_stationary():
    c = Ar1Corruptor(n_envs=1000, dt_env=DT, sigma_deg=5.0, tau_s=60.0,
                     seed=0)
    c.reset(np.zeros(1000))
    first = c.corrupt(np.zeros(1000))
    # The FIRST post-reset sample is already stationary (no cold start at 0).
    assert abs(first.std() - 5.0) < 0.5


def test_lag_exact_shift_on_ramp():
    lag_s = 30.0  # = 3 env steps at dt_env=10
    c = LagCorruptor(n_envs=2, dt_env=DT, lag_s=lag_s)
    wd0 = np.array([270.0, 270.0])
    c.reset(wd0)
    trace = [wd0 + 1.0 * k for k in range(20)]  # 1 deg per step ramp
    fed = [c.corrupt(w) for w in trace]
    # Pre-fill: the first lag_steps outputs hold wd0.
    for k in range(3):
        np.testing.assert_allclose(fed[k], wd0)
    # After that: exact 3-step shift.
    for k in range(3, 20):
        np.testing.assert_allclose(fed[k], trace[k - 3])


def test_lag_zero_is_identity():
    c = LagCorruptor(n_envs=2, dt_env=DT, lag_s=0.0)
    c.reset(np.array([270.0, 280.0]))
    wd = np.array([300.0, 310.0])
    np.testing.assert_allclose(c.corrupt(wd), wd)


def test_lag_requires_reset():
    c = LagCorruptor(n_envs=2, dt_env=DT, lag_s=30.0)
    with pytest.raises(RuntimeError):
        c.corrupt(np.zeros(2))


def test_make_corruptor_dispatch_and_errors():
    assert type(make_corruptor("none", 2, DT)) is WdCorruptor
    assert isinstance(make_corruptor("bias:-2", 2, DT), BiasCorruptor)
    assert isinstance(make_corruptor("ar1:sigma=5,tau=60", 2, DT),
                      Ar1Corruptor)
    assert isinstance(make_corruptor("lag:60", 2, DT), LagCorruptor)
    for bad in ("gauss:3", "ar1:sigma=5", "ar1:sigma=5,tau=0",
                "ar1:sigma=5,tau=60,rho=0.5", "lag:-10", "bias:x"):
        with pytest.raises((ValueError, KeyError)):
            make_corruptor(bad, 2, DT)


def test_ar1_seed_reproducible():
    a = Ar1Corruptor(2, DT, 5.0, 60.0, seed=42)
    b = Ar1Corruptor(2, DT, 5.0, 60.0, seed=42)
    wd = np.zeros(2)
    a.reset(wd)
    b.reset(wd)
    for _ in range(10):
        np.testing.assert_array_equal(a.corrupt(wd), b.corrupt(wd))
