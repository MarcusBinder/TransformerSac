"""Tests for the wind-direction schedule registries (helpers/wd_functions.py).

Two registries live side by side and MUST stay disjoint:

- ``WD_FUNCTIONS``      -- deterministic EVAL schedules, 1-arg ``f(t) -> wd``.
- ``TRAIN_WD_FACTORIES`` -- randomized TRAINING schedules, 2-arg ``f(t, base_wd) -> wd``.

The training family is *relative* (``wd(0) == base_wd``) so it composes with the
env's per-episode wd domain randomization instead of replacing it, and it
re-draws its episode parameters when called at ``t == 0.0`` -- the hook that
``wind_manager.make_wind_direction_list`` gives us, since it rebuilds the whole
wd list once per reset always starting from t=0.
"""

import numpy as np
import pytest

from helpers.wd_functions import (
    WD_FUNCTIONS,
    TRAIN_WD_FACTORIES,
    get_train_wd_factory,
    get_wd_function,
    make_dr_ramp,
    step_ramp_270_315,
)

# The env queries the schedule once per sim step; dt_sim = 10 s in every arm of
# the change_wd_2 sweep, and an episode is 500 env steps x dt_env 10 = 5000 s.
DT_SIM = 10.0
EPISODE_SECONDS = 5000.0
EPISODE_GRID = np.arange(0.0, EPISODE_SECONDS + DT_SIM, DT_SIM)

TRAIN_NAMES = sorted(TRAIN_WD_FACTORIES)


def _trajectory(fn, base_wd=270.0, grid=EPISODE_GRID):
    """Sample a schedule the way make_wind_direction_list does: t ascending from 0."""
    return np.array([float(fn(float(t), base_wd)) for t in grid])


# ---------------------------------------------------------------------------
# Registry hygiene / leakage guard
# ---------------------------------------------------------------------------

def test_train_and_eval_registries_are_disjoint():
    assert set(WD_FUNCTIONS) & set(TRAIN_WD_FACTORIES) == set()


def test_eval_schedule_name_is_rejected_by_the_train_registry():
    with pytest.raises(ValueError, match="step_ramp_270_315"):
        get_train_wd_factory("step_ramp_270_315", seed=0)


def test_train_schedule_name_is_rejected_by_the_eval_registry():
    with pytest.raises(ValueError, match="dr_ramp"):
        get_wd_function("dr_ramp")


@pytest.mark.parametrize("name", TRAIN_NAMES)
def test_get_train_wd_factory_returns_a_callable(name):
    fn = get_train_wd_factory(name, seed=0)
    assert callable(fn)


# ---------------------------------------------------------------------------
# Relative schedules: no jump at the burn-in boundary
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", TRAIN_NAMES)
@pytest.mark.parametrize("base_wd", [225.0, 250.0, 270.0, 300.0, 315.0])
def test_delta_at_t0_is_zero(name, base_wd):
    """wd(0) must equal base_wd -- Phase 1 of the wd list holds base_wd."""
    fn = get_train_wd_factory(name, seed=7)
    assert float(fn(0.0, base_wd)) == pytest.approx(base_wd)


# ---------------------------------------------------------------------------
# Re-draw semantics
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", TRAIN_NAMES)
def test_consecutive_episodes_differ(name):
    fn = get_train_wd_factory(name, seed=3)
    first = _trajectory(fn)
    second = _trajectory(fn)
    assert not np.allclose(first, second)


@pytest.mark.parametrize("name", TRAIN_NAMES)
def test_redraw_fires_only_at_t0(name):
    """Re-sampling the same episode without touching t=0 must be deterministic."""
    fn = get_train_wd_factory(name, seed=3)
    _trajectory(fn)  # draw episode 1 (grid starts at 0)

    tail = EPISODE_GRID[1:]
    first = _trajectory(fn, grid=tail)
    second = _trajectory(fn, grid=tail)
    np.testing.assert_allclose(first, second)


@pytest.mark.parametrize("name", TRAIN_NAMES)
def test_same_seed_gives_identical_trajectories(name):
    a = _trajectory(get_train_wd_factory(name, seed=11))
    b = _trajectory(get_train_wd_factory(name, seed=11))
    np.testing.assert_allclose(a, b)


@pytest.mark.parametrize("name", TRAIN_NAMES)
def test_different_seeds_give_different_trajectories(name):
    """Each of the 30 vector envs is seeded args.seed + i; they must not share a walk."""
    a = _trajectory(get_train_wd_factory(name, seed=11))
    b = _trajectory(get_train_wd_factory(name, seed=12))
    assert not np.allclose(a, b)


# ---------------------------------------------------------------------------
# Range and rate bounds
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", TRAIN_NAMES)
def test_wd_stays_inside_the_training_wd_domain(name):
    """hard_2 draws base_wd ~ U[225, 315]; the walk must not leave that band."""
    fn = get_train_wd_factory(name, seed=5)
    for base_wd in (225.0, 245.0, 270.0, 295.0, 315.0):
        for _ in range(20):
            wd = _trajectory(fn, base_wd=base_wd)
            assert wd.min() >= 225.0 - 1e-6
            assert wd.max() <= 315.0 + 1e-6


@pytest.mark.parametrize(
    "name, rate_lo, rate_hi",
    [("dr_ramp", 0.02, 0.25), ("dr_ramp_narrow", 0.09, 0.14)],
)
def test_instantaneous_rate_never_exceeds_the_declared_band(name, rate_lo, rate_hi):
    fn = get_train_wd_factory(name, seed=5)
    for _ in range(50):
        wd = _trajectory(fn)
        rates = np.abs(np.diff(wd)) / DT_SIM
        assert rates.max() <= rate_hi + 1e-6


@pytest.mark.parametrize(
    "name, rate_lo, rate_hi",
    [("dr_ramp", 0.02, 0.25), ("dr_ramp_narrow", 0.09, 0.14)],
)
def test_every_episode_actually_ramps(name, rate_lo, rate_hi):
    """A hold-only episode would silently turn the arm into a static-wd control."""
    fn = get_train_wd_factory(name, seed=5)
    for _ in range(50):
        wd = _trajectory(fn)
        rates = np.abs(np.diff(wd)) / DT_SIM
        assert rates.max() >= rate_lo - 1e-6


@pytest.mark.parametrize("name", TRAIN_NAMES)
def test_schedule_keeps_moving_across_the_whole_episode(name):
    """The eval ramp ends at t=1000 s but a training episode is 5000 s long; a
    single ramp would leave 80 % of the episode static."""
    fn = get_train_wd_factory(name, seed=5)
    for _ in range(20):
        wd = _trajectory(fn)
        late = wd[EPISODE_GRID >= 2000.0]
        assert np.abs(np.diff(late)).max() > 0.0


# ---------------------------------------------------------------------------
# Leakage: the training family must never emit the eval instance
# ---------------------------------------------------------------------------

def test_dr_ramp_narrow_never_reproduces_the_eval_schedule():
    eval_grid = np.arange(0.0, 1000.0 + DT_SIM, DT_SIM)
    target = np.asarray(step_ramp_270_315(eval_grid), dtype=float)

    fn = get_train_wd_factory("dr_ramp_narrow", seed=0)
    for _ in range(10_000):
        wd = _trajectory(fn, base_wd=270.0, grid=eval_grid)
        assert not np.allclose(wd, target, atol=0.5)


# ---------------------------------------------------------------------------
# make_dr_ramp directly
# ---------------------------------------------------------------------------

def test_make_dr_ramp_honours_custom_wd_bounds():
    fn = make_dr_ramp(seed=1, rate_lo=0.05, rate_hi=0.2, exc_lo=10.0, exc_hi=30.0,
                      wd_lo=260.0, wd_hi=280.0)
    for _ in range(20):
        wd = _trajectory(fn, base_wd=270.0)
        assert wd.min() >= 260.0 - 1e-6
        assert wd.max() <= 280.0 + 1e-6
