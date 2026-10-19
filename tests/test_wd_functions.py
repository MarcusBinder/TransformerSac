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
    ABSOLUTE_TRAIN_NAMES,
    WD_FUNCTIONS,
    TRAIN_WD_FACTORIES,
    build_eval_specs,
    get_train_wd_factory,
    get_wd_function,
    make_cycle,
    make_cycle_knots,
    make_cycle_train,
    make_dr_ramp,
    make_hold_ramp,
    make_richards_segment,
    make_static,
    parse_eval_ws,
    static_235,
    static_315,
    step_ramp_270_315,
)

# The env queries the schedule once per sim step; dt_sim = 10 s in every arm of
# the change_wd_2 sweep, and an episode is 500 env steps x dt_env 10 = 5000 s.
DT_SIM = 10.0
EPISODE_SECONDS = 5000.0
EPISODE_GRID = np.arange(0.0, EPISODE_SECONDS + DT_SIM, DT_SIM)

TRAIN_NAMES = sorted(TRAIN_WD_FACTORIES)
# Relative (f(t, base_wd)) vs absolute (f(t)) training families.
RELATIVE_TRAIN_NAMES = sorted(set(TRAIN_WD_FACTORIES) - set(ABSOLUTE_TRAIN_NAMES))
ABSOLUTE_NAMES = sorted(ABSOLUTE_TRAIN_NAMES)
# Stage 14: training schedules that hold base_wd for a random FRACTION of their
# episodes (mix_static_ramp, p_static 0.7). Exempt from the "every episode
# moves / consecutive episodes differ" tests; they keep the domain, redraw and
# seed tests and get their own static-fraction / rate test below.
# Stage 15: mix_static_richards (p_static 0.5) holds base_wd on half its episodes too.
STATIC_EPISODE_TRAIN_NAMES = frozenset({"mix_static_ramp", "mix_static_richards"})
MOVING_TRAIN_NAMES = [n for n in TRAIN_NAMES if n not in STATIC_EPISODE_TRAIN_NAMES]


def _takes_base_wd(fn) -> bool:
    return getattr(fn, "takes_base_wd", True)


def _trajectory(fn, base_wd=270.0, grid=EPISODE_GRID):
    """Sample a schedule the way make_wind_direction_list does: t ascending from 0.

    Dispatches on the schedule family: relative schedules get ``base_wd``,
    absolute ones (``takes_base_wd = False``) are called with ``t`` alone.
    """
    if _takes_base_wd(fn):
        return np.array([float(fn(float(t), base_wd)) for t in grid])
    return np.array([float(fn(float(t))) for t in grid])


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

@pytest.mark.parametrize("name", RELATIVE_TRAIN_NAMES)
@pytest.mark.parametrize("base_wd", [225.0, 250.0, 270.0, 300.0, 315.0])
def test_delta_at_t0_is_zero(name, base_wd):
    """wd(0) must equal base_wd -- Phase 1 of the wd list holds base_wd."""
    fn = get_train_wd_factory(name, seed=7)
    assert float(fn(0.0, base_wd)) == pytest.approx(base_wd)


# ---------------------------------------------------------------------------
# Re-draw semantics
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", MOVING_TRAIN_NAMES)
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
    [("dr_ramp", 0.02, 0.25), ("dr_ramp_narrow", 0.09, 0.14),
     ("dr_ramp_les", 0.03, 0.08), ("cycle_270_235_305_slow_phase", 35.0 / 320.0, 35.0 / 320.0),
     # Stage 15 (cell R): Emily's Richards shape peaks at 0.0336 deg/s (tau 322, nu 0.39);
     # the DR cycle spans tau 160-640 x nu 0.25-1.0 -> 0.0149-0.0727 deg/s (see the
     # Stage-15 block below); the stair steps are 11-12 deg over 340 s.
     ("richards_270_235_phase", 0.0336 - 2e-3, 0.0336 + 2e-3),
     ("richards_270_235_dr_phase", 0.014, 0.074),
     ("mix_cycle_richards_phase", 0.014, 35.0 / 320.0),
     ("cycle_stair_phase", 11.0 / 340.0 - 1e-4, 12.0 / 340.0 + 1e-4)],
)
def test_instantaneous_rate_never_exceeds_the_declared_band(name, rate_lo, rate_hi):
    fn = get_train_wd_factory(name, seed=5)
    for _ in range(50):
        wd = _trajectory(fn)
        rates = np.abs(np.diff(wd)) / DT_SIM
        assert rates.max() <= rate_hi + 1e-6


@pytest.mark.parametrize(
    "name, rate_lo, rate_hi",
    [("dr_ramp", 0.02, 0.25), ("dr_ramp_narrow", 0.09, 0.14),
     ("dr_ramp_les", 0.03, 0.08), ("cycle_270_235_305_slow_phase", 35.0 / 320.0, 35.0 / 320.0),
     # Stage 15 (cell R): Emily's Richards shape peaks at 0.0336 deg/s (tau 322, nu 0.39);
     # the DR cycle spans tau 160-640 x nu 0.25-1.0 -> 0.0149-0.0727 deg/s (see the
     # Stage-15 block below); the stair steps are 11-12 deg over 340 s.
     ("richards_270_235_phase", 0.0336 - 2e-3, 0.0336 + 2e-3),
     ("richards_270_235_dr_phase", 0.014, 0.074),
     ("mix_cycle_richards_phase", 0.014, 35.0 / 320.0),
     ("cycle_stair_phase", 11.0 / 340.0 - 1e-4, 12.0 / 340.0 + 1e-4)],
)
def test_every_episode_actually_ramps(name, rate_lo, rate_hi):
    """A hold-only episode would silently turn the arm into a static-wd control."""
    fn = get_train_wd_factory(name, seed=5)
    for _ in range(50):
        wd = _trajectory(fn)
        rates = np.abs(np.diff(wd)) / DT_SIM
        assert rates.max() >= rate_lo - 1e-6


@pytest.mark.parametrize("name", MOVING_TRAIN_NAMES)
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
# static_315 -- the control change_wd_2 was missing (change_wd_3)
# ---------------------------------------------------------------------------

def test_static_315_is_constant_for_scalar_t():
    for t in (0.0, 1.0, 300.0, 1000.0, 5000.0):
        assert float(static_315(t)) == pytest.approx(315.0)


def test_static_315_is_constant_for_array_t():
    wd = static_315(EPISODE_GRID)
    assert wd.shape == EPISODE_GRID.shape
    np.testing.assert_allclose(wd, 315.0)


def test_static_315_matches_the_ramp_endpoint():
    """The point of the control: it must sit exactly where the ramp settles."""
    assert float(static_315(0.0)) == pytest.approx(float(step_ramp_270_315(1000.0)))


def test_static_315_is_registered_as_an_eval_schedule():
    assert WD_FUNCTIONS["static_315"] is static_315
    assert get_wd_function("static_315") is static_315


def test_static_315_is_not_a_training_schedule():
    """The leakage guard: eval schedules must never be selectable for training."""
    assert "static_315" not in TRAIN_WD_FACTORIES
    with pytest.raises(ValueError, match="static_315"):
        get_train_wd_factory("static_315", seed=0)


# ---------------------------------------------------------------------------
# Eval ladder: (wd schedule x eval ws) cross product and namespace back-compat
# ---------------------------------------------------------------------------

# The exact string change_wd_2 passed. Its key namespace is the compatibility
# target: change_wd_3 must not move these keys, or the two sweeps stop being
# comparable with no error raised anywhere.
CHWD2_SCHEDULES = "step_ramp_270_315,static_270"


@pytest.mark.parametrize("raw, expected", [
    ("12", [12.0]),
    ("12,10.5", [12.0, 10.5]),
    (" 12 , 10.5 ", [12.0, 10.5]),
    ("12,10.5,", [12.0, 10.5]),
    (12, [12.0]),
    (None, [12.0]),
])
def test_parse_eval_ws(raw, expected):
    assert parse_eval_ws(raw) == expected


def test_no_eval_wd_function_means_no_specs():
    """Falls back to the single static-wd evaluator, which pins no wind at all."""
    assert build_eval_specs(None, "12,10.5") == []
    assert build_eval_specs("", "12") == []
    assert build_eval_specs("  ", "12") == []


def test_single_ws_namespace_is_byte_identical_to_change_wd_2():
    specs = build_eval_specs(CHWD2_SCHEDULES, "12")
    assert [name for _, _, name in specs] == ["step_ramp_270_315", "static_270"]
    assert [ws for _, ws, _ in specs] == [12.0, 12.0]


def test_multi_ws_builds_the_full_cross_product():
    specs = build_eval_specs("step_ramp_270_315,static_270,static_315", "12,10.5")
    assert len(specs) == 6
    assert [name for _, _, name in specs] == [
        "step_ramp_270_315/ws12", "step_ramp_270_315/ws10.5",
        "static_270/ws12", "static_270/ws10.5",
        "static_315/ws12", "static_315/ws10.5",
    ]


def test_first_spec_owns_the_unprefixed_keys():
    """Spec 0 is (first schedule, first speed) -- the i == 0 branch in
    run_all_evaluations writes the historical unprefixed eval/... keys from it."""
    specs = build_eval_specs("step_ramp_270_315,static_270,static_315", "12,10.5")
    wd, ws, _ = specs[0]
    assert (wd, ws) == ("step_ramp_270_315", 12.0)


def test_unknown_schedule_fails_before_any_env_is_built():
    with pytest.raises(ValueError, match="static_999"):
        build_eval_specs("static_999", "12")


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


# ---------------------------------------------------------------------------
# LES-3x3 scenario family (make_hold_ramp + static_235) -- endpoint/rate
# asserts mirroring the static_315 block
# ---------------------------------------------------------------------------

LES_RAMP_RATE = 35.0 / 600.0  # deg/s of the target scenario's transition


def test_hold_ramp_270_235_endpoints_and_rate():
    fn = get_wd_function("hold_ramp_270_235")
    # Holds: exactly 270 through the pre-hold, exactly 235 after the ramp.
    assert float(fn(0.0)) == pytest.approx(270.0)
    assert float(fn(2044.9)) == pytest.approx(270.0)
    assert float(fn(2045.0 + 600.0)) == pytest.approx(235.0)
    assert float(fn(10_000.0)) == pytest.approx(235.0)
    # Ramp midpoint and constant rate.
    assert float(fn(2045.0 + 300.0)) == pytest.approx(252.5)
    t = np.arange(2045.0, 2045.0 + 600.0 + 1.0, 1.0)
    rates = np.diff(np.asarray(fn(t), dtype=float))
    np.testing.assert_allclose(rates, -LES_RAMP_RATE, atol=1e-9)


def test_hold_ramp_270_235_short_is_the_same_ramp_moved_earlier():
    long = get_wd_function("hold_ramp_270_235")
    short = get_wd_function("hold_ramp_270_235_short")
    # Identical trajectory once the pre-holds are aligned (300 s vs 2045 s).
    t = np.arange(0.0, 1500.0, 10.0)
    np.testing.assert_allclose(
        np.asarray(short(t), dtype=float),
        np.asarray(long(t + (2045.0 - 300.0)), dtype=float),
    )
    # And it fits the 140-step in-training eval: settled well before 1400 s.
    assert float(short(900.0)) == pytest.approx(235.0)


def test_hold_ramp_270_305_is_the_rate_matched_veer_mirror():
    down = get_wd_function("hold_ramp_270_235")
    up = get_wd_function("hold_ramp_270_305")
    t = EPISODE_GRID
    d_down = np.asarray(down(t), dtype=float) - 270.0
    d_up = np.asarray(up(t), dtype=float) - 270.0
    np.testing.assert_allclose(d_up, -d_down)  # same timing, opposite sign


def test_hold_ramp_schedules_accept_scalar_and_array_t():
    fn = get_wd_function("hold_ramp_270_235")
    scalar = float(fn(2345.0))
    arr = np.asarray(fn(np.array([2345.0])), dtype=float)
    assert scalar == pytest.approx(arr[0])


def test_hold_ramp_270_315_is_untouched_by_the_factory():
    """T3 provenance guard: the legacy schedule stays verbatim (600/1200 shape),
    NOT a make_hold_ramp instance (2045/600 shape)."""
    legacy = get_wd_function("hold_ramp_270_315")
    assert float(legacy(0.0)) == pytest.approx(270.0)
    assert float(legacy(600.0 + 1200.0)) == pytest.approx(315.0)
    assert float(legacy(1200.0)) == pytest.approx(270.0 + 45.0 * (600.0 / 1200.0))


def test_static_235_is_constant_and_matches_the_ramp_endpoint():
    ramp = get_wd_function("hold_ramp_270_235")
    for t in (0.0, 1.0, 300.0, 1000.0, 5000.0):
        assert float(static_235(t)) == pytest.approx(235.0)
    wd = static_235(EPISODE_GRID)
    assert wd.shape == EPISODE_GRID.shape
    np.testing.assert_allclose(wd, 235.0)
    assert float(static_235(0.0)) == pytest.approx(float(ramp(5000.0)))


def test_les_schedules_are_registered_as_eval_not_train():
    for name in ("hold_ramp_270_235", "hold_ramp_270_235_short",
                 "hold_ramp_270_305", "static_235"):
        assert callable(get_wd_function(name))
        assert name not in TRAIN_WD_FACTORIES
    assert "dr_ramp_les" in TRAIN_WD_FACTORIES
    assert "dr_ramp_les" not in WD_FUNCTIONS


def test_dr_ramp_les_stays_below_the_mtm12_frame_slew_limit():
    """The design invariant of the campaign: EVERY dr_ramp_les rate must be
    trackable by the les_3x3 frame at max_turb_move=12 (slew limit
    12*360/(2*pi*1533.8)/5 = 0.0897 deg/s), so wd_small == 0 throughout
    training and env.wd is exact and causal."""
    frame_slew = 12.0 * 360.0 / (2.0 * np.pi * 1533.8) / 5.0
    fn = get_train_wd_factory("dr_ramp_les", seed=5)
    for _ in range(50):
        wd = _trajectory(fn)
        rates = np.abs(np.diff(wd)) / DT_SIM
        assert rates.max() < frame_slew


# ---------------------------------------------------------------------------
# LES-3x3 Stage 3: narrow-band retreat (les_recipe_nb, wd in [265, 275])
# ---------------------------------------------------------------------------

NB_RAMP_RATE = 5.0 / 300.0  # deg/s of the in-band 5 deg / 300 s ramps


def test_stage3_schedules_are_registered_as_eval_not_train():
    for name in ("static_265", "static_275", "hold_ramp_270_275",
                 "hold_ramp_270_265", "hold_ramp_270_275_short"):
        assert callable(get_wd_function(name))
        assert name not in TRAIN_WD_FACTORIES


def test_make_static_is_constant_and_named():
    fn = make_static(265.0)
    assert fn.__name__ == "static_265"
    wd = fn(EPISODE_GRID)
    assert wd.shape == EPISODE_GRID.shape
    np.testing.assert_allclose(wd, 265.0)
    assert float(get_wd_function("static_275")(4321.0)) == pytest.approx(275.0)
    assert float(get_wd_function("static_265")(0.0)) == pytest.approx(265.0)


@pytest.mark.parametrize("name,wd_to", [("hold_ramp_270_275", 275.0),
                                        ("hold_ramp_270_265", 265.0)])
def test_nb_hold_ramps_endpoints_rate_and_band(name, wd_to):
    fn = get_wd_function(name)
    assert float(fn(0.0)) == pytest.approx(270.0)
    assert float(fn(2044.0)) == pytest.approx(270.0)
    assert float(fn(2045.0 + 300.0)) == pytest.approx(wd_to)
    assert float(fn(5000.0)) == pytest.approx(wd_to)
    wd = fn(EPISODE_GRID)
    rates = np.abs(np.diff(wd)) / DT_SIM
    assert rates.max() == pytest.approx(NB_RAMP_RATE)
    # Whole trajectory stays inside the les_recipe_nb training band.
    assert wd.min() >= 265.0 - 1e-9 and wd.max() <= 275.0 + 1e-9
    # Static endpoint controls match the ramp endpoints exactly.
    assert float(get_wd_function(f"static_{wd_to:g}")(0.0)) == pytest.approx(float(fn(5000.0)))
    # 30 env steps (dt_env 10) of ramp -> windowable, unlike a 5 deg ramp at
    # the 0.0583 deg/s scenario rate (~9 steps).
    assert 300.0 / 10.0 == 30


def test_hold_ramp_270_275_short_is_the_same_ramp_moved_earlier():
    long = get_wd_function("hold_ramp_270_275")
    short = get_wd_function("hold_ramp_270_275_short")
    t = np.arange(0.0, 300.0 + 300.0 + 200.0, DT_SIM)
    np.testing.assert_allclose(short(t), long(t + (2045.0 - 300.0)))
    # Fits the 140-step (1400 s) in-training eval window with a trailing hold.
    assert float(short(1399.0)) == pytest.approx(275.0)


# ---------------------------------------------------------------------------
# LES-3x3 Stage 5: train ON the transition -- cyclic 270 <-> 235 schedules
# ---------------------------------------------------------------------------

CYC_RATE = 35.0 / 200.0  # 0.175 deg/s (200 s ramps)
LES3X3_MAX_DIST = 1533.8  # m, max pairwise distance on les_3x3
DT_SIM_LES = 5.0


def _frame_slew(mtm: float) -> float:
    """Mean-flow frame slew limit on les_3x3 (deg/s) for max_turb_move=mtm."""
    return mtm * 360.0 / (2.0 * np.pi * LES3X3_MAX_DIST) / DT_SIM_LES


def test_cycle_270_235_shape_period_and_rate():
    fn = get_wd_function("cycle_270_235")
    assert fn.period == pytest.approx(2400.0)
    # hold1 / ramp / hold2 / ramp_up within one period
    assert float(fn(0.0)) == pytest.approx(270.0)
    assert float(fn(999.0)) == pytest.approx(270.0)
    assert float(fn(1100.0)) == pytest.approx(252.5)  # ramp midpoint
    assert float(fn(1200.0)) == pytest.approx(235.0)
    assert float(fn(2199.0)) == pytest.approx(235.0)
    assert float(fn(2300.0)) == pytest.approx(252.5)  # up-ramp midpoint
    assert float(fn(2400.0)) == pytest.approx(270.0)  # period closes
    # periodicity
    t = np.arange(0.0, 2400.0, 1.0)
    np.testing.assert_allclose(fn(t), fn(t + 2400.0))
    np.testing.assert_allclose(fn(t), fn(t + 3 * 2400.0))
    # rate is exactly +-0.175 deg/s on the ramps, 0 on the holds
    rates = np.diff(np.asarray(fn(np.arange(0.0, 4800.0 + 1.0, 1.0)), dtype=float))
    assert np.abs(rates).max() == pytest.approx(CYC_RATE)
    assert set(np.round(np.abs(rates), 9)) == {0.0, round(CYC_RATE, 9)}
    # range
    assert fn(t).min() == pytest.approx(235.0) and fn(t).max() == pytest.approx(270.0)


def test_cycle_accepts_scalar_and_array_t_and_phase_shifts():
    fn = get_wd_function("cycle_270_235")
    assert float(fn(1100.0)) == pytest.approx(np.asarray(fn(np.array([1100.0])))[0])
    shifted = make_cycle(270.0, 235.0, 1000.0, 200.0, phase=700.0)
    t = np.arange(0.0, 5000.0, 10.0)
    np.testing.assert_allclose(shifted(t), fn(t + 700.0))


def test_cycle_270_235_short_fits_the_140_step_eval_window():
    short = get_wd_function("cycle_270_235_short")
    assert short.period == pytest.approx(1400.0)  # == 140 steps x dt_env 10
    assert float(short(0.0)) == pytest.approx(270.0)
    assert float(short(600.0)) == pytest.approx(252.5)
    assert float(short(700.0)) == pytest.approx(235.0)
    assert float(short(1199.0)) == pytest.approx(235.0)
    assert float(short(1400.0)) == pytest.approx(270.0)
    rates = np.diff(np.asarray(short(np.arange(0.0, 1400.0 + 1.0, 1.0)), dtype=float))
    assert np.abs(rates).max() == pytest.approx(CYC_RATE)


def test_cycle_schedules_are_registered_as_eval_not_train():
    for name in ("cycle_270_235", "cycle_270_235_short"):
        assert callable(get_wd_function(name))
        assert name not in TRAIN_WD_FACTORIES
    assert "cycle_270_235_phase" in TRAIN_WD_FACTORIES
    assert "cycle_270_235_phase" not in WD_FUNCTIONS
    assert ABSOLUTE_TRAIN_NAMES <= set(TRAIN_WD_FACTORIES)
    assert set(ABSOLUTE_TRAIN_NAMES) & set(WD_FUNCTIONS) == set()


def test_cycle_train_schedule_is_absolute_for_the_wind_manager():
    """The env classifies the schedule by arity: one positional arg => absolute."""
    from WindGym.core.wind_manager import WindManager
    fn = get_train_wd_factory("cycle_270_235_phase", seed=0)
    assert fn.takes_base_wd is False
    assert WindManager._wd_function_takes_base_wd(fn) is False
    # and every relative schedule still reads as relative
    for name in RELATIVE_TRAIN_NAMES:
        assert WindManager._wd_function_takes_base_wd(get_train_wd_factory(name, seed=0)) is True


def test_cycle_train_starts_in_the_270_hold_every_episode_and_phases_vary():
    """Phase rule: phase ~ U[0, t_hold) so f(0) == 270 == the pinned burn-in wd
    (no jump at t=0), while the first ramp starts anywhere in [0, 1000] s."""
    fn = get_train_wd_factory("cycle_270_235_phase", seed=5)
    base = get_wd_function("cycle_270_235")
    phases = []
    for _ in range(100):
        wd = _trajectory(fn)
        assert wd[0] == pytest.approx(270.0)
        phase = fn.phase
        assert 0.0 <= phase < 1000.0
        phases.append(phase)
        # the whole episode is the fixed cycle shifted by that phase
        np.testing.assert_allclose(wd, np.asarray(base(EPISODE_GRID + phase), dtype=float))
        # first ramp starts within the first 1000 s
        first_move = EPISODE_GRID[np.nonzero(np.abs(wd - 270.0) > 1e-9)[0][0]]
        assert first_move <= 1000.0 + DT_SIM
    phases = np.asarray(phases)
    assert phases.std() > 100.0  # genuinely spread over [0, 1000)
    assert len(np.unique(np.round(phases, 3))) > 90


def test_cycle_train_phase_is_drawn_only_at_t0():
    fn = get_train_wd_factory("cycle_270_235_phase", seed=5)
    _trajectory(fn)
    p = fn.phase
    _trajectory(fn, grid=EPISODE_GRID[1:])
    assert fn.phase == p
    _trajectory(fn)
    assert fn.phase != p


def test_cycle_train_phase_max_must_stay_inside_the_first_hold():
    with pytest.raises(ValueError, match="phase_max"):
        make_cycle_train(270.0, 235.0, 1000.0, 200.0, seed=0, phase_max=1200.0)
    # exactly t_hold is allowed (U[0, t_hold) never reaches it)
    make_cycle_train(270.0, 235.0, 1000.0, 200.0, seed=0, phase_max=1000.0)


def test_cycle_rate_vs_frame_slew_limit():
    """0.175 deg/s ramp: NOT trackable at the Stage-2/4 mtm=12 (0.0897 deg/s),
    trackable at the Stage-5 choice mtm=25 (0.1868 deg/s). 23.4 is the threshold."""
    assert CYC_RATE > _frame_slew(12.0)
    assert CYC_RATE > _frame_slew(23.0)
    assert CYC_RATE < _frame_slew(24.0)
    assert CYC_RATE < _frame_slew(25.0)
    fn = get_train_wd_factory("cycle_270_235_phase", seed=5)
    for _ in range(10):
        wd = _trajectory(fn)
        rates = np.abs(np.diff(wd)) / DT_SIM
        assert rates.max() == pytest.approx(CYC_RATE)
        assert rates.max() < _frame_slew(25.0)


def test_cycle_train_stays_inside_the_band235_control_band():
    fn = get_train_wd_factory("cycle_270_235_phase", seed=1)
    for _ in range(10):
        wd = _trajectory(fn)
        assert wd.min() >= 235.0 - 1e-9 and wd.max() <= 270.0 + 1e-9


# ---------------------------------------------------------------------------
# LES-3x3 Stage 6: IEA 22 MW "real" case -- the _slow (1.6x-scaled) family
# ---------------------------------------------------------------------------

SLOW_RATE = 35.0 / 320.0  # 0.109375 deg/s (320 s ramps)
# Max turbine distance from the bbox center of les_3x3 at D=284 m (IEA22):
# spacing 7D x 5D = 1988 x 1420 m -> half-diagonal sqrt(1988^2 + 1420^2).
IEA22_MAX_DIST = 2443.06  # m


def _frame_slew_iea22(mtm: float) -> float:
    """Mean-flow frame slew limit on the IEA22 les_3x3 (deg/s) at max_turb_move=mtm."""
    return mtm * 360.0 / (2.0 * np.pi * IEA22_MAX_DIST) / DT_SIM_LES


def test_cycle_270_235_slow_shape_period_and_rate():
    fn = get_wd_function("cycle_270_235_slow")
    assert fn.period == pytest.approx(3840.0)
    # hold1 / ramp / hold2 / ramp_up within one period
    assert float(fn(0.0)) == pytest.approx(270.0)
    assert float(fn(1599.0)) == pytest.approx(270.0)
    assert float(fn(1760.0)) == pytest.approx(252.5)  # ramp midpoint
    assert float(fn(1920.0)) == pytest.approx(235.0)
    assert float(fn(3519.0)) == pytest.approx(235.0)
    assert float(fn(3680.0)) == pytest.approx(252.5)  # up-ramp midpoint
    assert float(fn(3840.0)) == pytest.approx(270.0)  # period closes
    # periodicity
    t = np.arange(0.0, 3840.0, 1.0)
    np.testing.assert_allclose(fn(t), fn(t + 3840.0))
    np.testing.assert_allclose(fn(t), fn(t + 3 * 3840.0))
    # rate is exactly +-0.109375 deg/s on the ramps, 0 on the holds
    rates = np.diff(np.asarray(fn(np.arange(0.0, 7680.0 + 1.0, 1.0)), dtype=float))
    assert np.abs(rates).max() == pytest.approx(SLOW_RATE)
    assert set(np.round(np.abs(rates), 9)) == {0.0, round(SLOW_RATE, 9)}
    # band
    assert fn(t).min() == pytest.approx(235.0) and fn(t).max() == pytest.approx(270.0)
    # two periods = 768 env steps @ dt_env=10 (the harvest cell length)
    assert 2 * fn.period / 10.0 == 768


def test_cycle_270_235_slow_short_fits_the_224_step_eval_window():
    short = get_wd_function("cycle_270_235_slow_short")
    assert short.period == pytest.approx(2240.0)  # == 224 steps x dt_env 10
    assert float(short(0.0)) == pytest.approx(270.0)
    assert float(short(960.0)) == pytest.approx(252.5)
    assert float(short(1120.0)) == pytest.approx(235.0)
    assert float(short(1919.0)) == pytest.approx(235.0)
    assert float(short(2240.0)) == pytest.approx(270.0)
    rates = np.diff(np.asarray(short(np.arange(0.0, 2240.0 + 1.0, 1.0)), dtype=float))
    assert np.abs(rates).max() == pytest.approx(SLOW_RATE)


def test_hold_ramp_270_235_slow_endpoints_and_rate():
    """The Stage-6 rate-transfer cell: 6 passthroughs of the IEA22 les_3x3 max
    pairwise extent (4886.1 m) at 9 m/s -> 3257 s pre-hold, then 35 deg over
    320 s. 470-step cell (4700 s) leaves 1123 s of trailing hold."""
    fn = get_wd_function("hold_ramp_270_235_slow")
    assert float(fn(0.0)) == pytest.approx(270.0)
    assert float(fn(3256.0)) == pytest.approx(270.0)
    assert float(fn(3257.0 + 160.0)) == pytest.approx(252.5)  # ramp midpoint
    assert float(fn(3257.0 + 320.0)) == pytest.approx(235.0)
    assert float(fn(4700.0)) == pytest.approx(235.0)
    grid = np.arange(0.0, 4700.0 + 1.0, 1.0)
    rates = np.abs(np.diff(np.asarray(fn(grid), dtype=float)))
    assert rates.max() == pytest.approx(SLOW_RATE)
    assert fn(grid).min() == pytest.approx(235.0)
    assert fn(grid).max() == pytest.approx(270.0)


def test_slow_schedules_are_registered_as_eval_not_train():
    for name in ("cycle_270_235_slow", "cycle_270_235_slow_short",
                 "hold_ramp_270_235_slow"):
        assert callable(get_wd_function(name))
        assert name not in TRAIN_WD_FACTORIES
    assert "cycle_270_235_slow_phase" in TRAIN_WD_FACTORIES
    assert "cycle_270_235_slow_phase" not in WD_FUNCTIONS
    assert "cycle_270_235_slow_phase" in ABSOLUTE_TRAIN_NAMES


def test_slow_cycle_train_is_absolute_starts_in_hold1_and_phases_vary():
    """Same phase rule as Stage 5: phase ~ U[0, t_hold=1600) so f(0) == 270 ==
    the pinned burn-in wd (les_recipe_pin270), no jump at t=0."""
    from WindGym.core.wind_manager import WindManager
    fn = get_train_wd_factory("cycle_270_235_slow_phase", seed=5)
    assert fn.takes_base_wd is False
    assert WindManager._wd_function_takes_base_wd(fn) is False
    base = get_wd_function("cycle_270_235_slow")
    phases = []
    for _ in range(100):
        wd = _trajectory(fn)
        assert wd[0] == pytest.approx(270.0)
        phase = fn.phase
        assert 0.0 <= phase < 1600.0
        phases.append(phase)
        # the whole episode is the fixed slow cycle shifted by that phase
        np.testing.assert_allclose(wd, np.asarray(base(EPISODE_GRID + phase), dtype=float))
        # first ramp starts within the first 1600 s
        first_move = EPISODE_GRID[np.nonzero(np.abs(wd - 270.0) > 1e-9)[0][0]]
        assert first_move <= 1600.0 + DT_SIM
        # band
        assert wd.min() >= 235.0 - 1e-9 and wd.max() <= 270.0 + 1e-9
    phases = np.asarray(phases)
    assert phases.std() > 160.0  # genuinely spread over [0, 1600)
    assert len(np.unique(np.round(phases, 3))) > 90


def test_slow_cycle_rate_vs_iea22_frame_slew_limit():
    """0.109375 deg/s ramp on the IEA22 farm: threshold mtm 23.32 -- mtm 23
    fails, 24/25/30 track (mtm 25 -> 7 % margin, campaign mtm 30 -> 29 %).
    The DTU constants (LES3X3_MAX_DIST, CYC_RATE thresholds) are untouched."""
    assert SLOW_RATE > _frame_slew_iea22(12.0)
    assert SLOW_RATE > _frame_slew_iea22(23.0)
    assert SLOW_RATE < _frame_slew_iea22(24.0)
    assert SLOW_RATE < _frame_slew_iea22(25.0)
    assert SLOW_RATE < _frame_slew_iea22(30.0)
    # margins quoted in the campaign notes
    assert _frame_slew_iea22(25.0) / SLOW_RATE == pytest.approx(1.072, abs=0.005)
    assert _frame_slew_iea22(30.0) / SLOW_RATE == pytest.approx(1.287, abs=0.005)
    # rate invariance: the _slow schedule is exactly trackable wherever the
    # Stage-5 cycle was at the DTU scale (both rate and farm scaled by 1.593x)
    fn = get_train_wd_factory("cycle_270_235_slow_phase", seed=5)
    for _ in range(10):
        wd = _trajectory(fn)
        rates = np.abs(np.diff(wd)) / DT_SIM
        assert rates.max() == pytest.approx(SLOW_RATE)
        assert rates.max() < _frame_slew_iea22(30.0)


# ---------------------------------------------------------------------------
# LES-3x3 Stage 12 (exp_12 `sym` / `symveer`): the two-sided _slow cycle
# 270 -> 235 -> 270 -> 305 -> 270 (knot-based make_cycle_knots)
# ---------------------------------------------------------------------------

SYM_LEVELS = (270.0, 235.0, 270.0, 305.0)
SYM_PERIOD = 4 * (1600.0 + 320.0)   # 7680 s
LONG_GRID = np.arange(0.0, 10000.0 + DT_SIM, DT_SIM)


def test_make_cycle_knots_with_two_levels_is_make_cycle():
    knots = make_cycle_knots((270.0, 235.0), 1600.0, 320.0)
    base = make_cycle(270.0, 235.0, 1600.0, 320.0)
    t = np.arange(0.0, 3 * 3840.0, 1.0)
    np.testing.assert_allclose(knots(t), base(t))
    assert knots.period == pytest.approx(base.period)
    assert float(knots(0.0)) == pytest.approx(270.0)
    # a phase shifts where t=0 lands, as in make_cycle
    np.testing.assert_allclose(make_cycle_knots((270.0, 235.0), 1600.0, 320.0, phase=500.0)(t), base(t + 500.0))


def test_make_cycle_knots_rejects_bad_timing():
    with pytest.raises(ValueError):
        make_cycle_knots(SYM_LEVELS, 1600.0, 0.0)
    with pytest.raises(ValueError):
        make_cycle_knots((270.0,), 1600.0, 320.0)


def test_cycle_270_235_305_slow_shape_period_rate_and_270_share():
    fn = get_wd_function("cycle_270_235_305_slow")
    assert fn.period == pytest.approx(SYM_PERIOD)
    for t, wd in ((0.0, 270.0), (1599.0, 270.0), (1760.0, 252.5), (1920.0, 235.0), (3519.0, 235.0),
                  (3680.0, 252.5), (3840.0, 270.0), (5439.0, 270.0), (5600.0, 287.5), (5760.0, 305.0),
                  (7359.0, 305.0), (7520.0, 287.5), (7680.0, 270.0)):
        assert float(fn(t)) == pytest.approx(wd), t
    t = np.arange(0.0, SYM_PERIOD, 1.0)
    np.testing.assert_allclose(fn(t), fn(t + SYM_PERIOD))
    rates = np.diff(np.asarray(fn(np.arange(0.0, 2 * SYM_PERIOD + 1.0, 1.0)), dtype=float))
    assert np.abs(rates).max() == pytest.approx(SLOW_RATE)
    assert set(np.round(np.abs(rates), 9)) == {0.0, round(SLOW_RATE, 9)}
    wd = np.asarray(fn(t), dtype=float)
    assert wd.min() == pytest.approx(235.0) and wd.max() == pytest.approx(305.0)
    # 42 % of the time at exactly 270 (two 1600 s holds of 7680 s), the rest split evenly
    share_270 = np.mean(np.isclose(wd, 270.0))
    assert share_270 == pytest.approx(3200.0 / SYM_PERIOD, abs=0.002)
    assert np.mean(wd > 270.0) == pytest.approx(np.mean(wd < 270.0), abs=0.002)


def test_cycle_270_235_305_slow_phase_is_absolute_starts_in_a_270_hold_and_phases_vary():
    from WindGym.core.wind_manager import WindManager
    fn = get_train_wd_factory("cycle_270_235_305_slow_phase", seed=5)
    assert fn.takes_base_wd is False
    assert WindManager._wd_function_takes_base_wd(fn) is False
    base = get_wd_function("cycle_270_235_305_slow")
    phases, first_dir = [], []
    for _ in range(200):
        wd = _trajectory(fn)
        assert wd[0] == pytest.approx(270.0)
        phase = fn.phase
        # either 270 hold: [0, 1600) or [3840, 5440)
        assert (0.0 <= phase < 1600.0) or (3840.0 <= phase < 5440.0)
        np.testing.assert_allclose(wd, np.asarray(base(EPISODE_GRID + phase), dtype=float))
        phases.append(phase)
        first = wd[np.nonzero(np.abs(wd - 270.0) > 1e-9)[0][0]]
        first_dir.append(np.sign(first - 270.0))
    phases = np.asarray(phases)
    assert (phases < 1600.0).sum() > 60 and (phases >= 3840.0).sum() > 60   # both holds are used
    assert len(np.unique(np.round(phases, 3))) > 180
    first_dir = np.asarray(first_dir)
    assert (first_dir < 0).sum() > 60 and (first_dir > 0).sum() > 60   # either excursion comes first


def test_cycle_270_235_305_slow_phase_visits_both_excursions_in_a_10000_s_episode():
    fn = get_train_wd_factory("cycle_270_235_305_slow_phase", seed=9)
    for _ in range(50):
        wd = _trajectory(fn, grid=LONG_GRID)
        assert wd.min() == pytest.approx(235.0) and wd.max() == pytest.approx(305.0)
        rates = np.abs(np.diff(wd)) / DT_SIM
        assert rates.max() == pytest.approx(SLOW_RATE)


def test_cycle_270_235_305_registries_are_disjoint_and_absolute():
    assert "cycle_270_235_305_slow" in WD_FUNCTIONS and "cycle_270_235_305_slow" not in TRAIN_WD_FACTORIES
    assert "cycle_270_235_305_slow_phase" in TRAIN_WD_FACTORIES and "cycle_270_235_305_slow_phase" not in WD_FUNCTIONS
    assert "cycle_270_235_305_slow_phase" in ABSOLUTE_TRAIN_NAMES
    assert set(ABSOLUTE_TRAIN_NAMES) & set(WD_FUNCTIONS) == set()
    with pytest.raises(ValueError):
        get_train_wd_factory("cycle_270_235_305_slow", seed=0)


# ---------------------------------------------------------------------------
# Stage 14: mix_static_ramp (70 % static episodes, 30 % dr_ramp-style walks)
# ---------------------------------------------------------------------------

def test_mix_static_ramp_static_fraction_and_rate():
    fn = get_train_wd_factory("mix_static_ramp", seed=7)
    n_static = 0
    for _ in range(400):
        tr = _trajectory(fn, base_wd=250.0)      # t == 0 first -> redraw
        if np.allclose(tr, 250.0):
            n_static += 1
        else:
            rate = np.abs(np.diff(tr)) / DT_SIM
            # frame slew limit at max_turb_move 30 on les_3x3 (0.1407 deg/s)
            assert rate.max() <= 0.1407 + 1e-9
            assert 225.0 - 1e-9 <= tr.min() and tr.max() <= 315.0 + 1e-9
    assert 0.6 <= n_static / 400 <= 0.8


def test_mix_static_ramp_is_relative_and_zero_at_t0():
    fn = get_train_wd_factory("mix_static_ramp", seed=3)
    assert "mix_static_ramp" not in ABSOLUTE_TRAIN_NAMES
    assert _takes_base_wd(fn)
    for base in (225.0, 270.0, 315.0):
        assert fn(0.0, base) == base


def test_mix_static_ramp_ramp_episodes_actually_ramp():
    """A ramp episode must reach the dr_ramp_narrow rate floor; the static
    fraction is the ONLY way an episode may sit still."""
    fn = get_train_wd_factory("mix_static_ramp", seed=9)
    for _ in range(100):
        tr = _trajectory(fn, base_wd=270.0)
        if not np.allclose(tr, 270.0):
            assert np.abs(np.diff(tr)).max() / DT_SIM >= 0.09 - 1e-6


# ---------------------------------------------------------------------------
# Stage 15 (exp_15, cell R): Emily's Richards wd profile as TRAINING schedules
# ---------------------------------------------------------------------------
# Cell R (2026-10-19): on Emily's LES profile richards:270:235:1690:322:0.39 (3600 s)
# no Stage-14 arm beat greedy; the loss sits in the first 10 deg of the slow turn
# (265 -> 255), which training (0.109-0.14 deg/s linear ramps) never dwelt in.
# The new schedules put the slow generalized-logistic turn (peak 0.0336 deg/s)
# into training: the exact Emily shape (richards_270_235_phase), a tau / nu / hold
# randomised version (richards_270_235_dr_phase), a 50/50 mixture with the R3
# linear cycle (mix_cycle_richards_phase), a linear stair with dwells at 258 / 246
# on both sides of the 250.3 deg wake-alignment crossing (cycle_stair_phase), and
# a RELATIVE static / Richards-walk mixture inside [235, 270] (mix_static_richards).

# Emily's profile, verbatim LESRL dwm_cli.make_richards(270, 235, 1690, 322, 0.39):
# wd(t) = 270 + (235 - 270) * (1 + exp(-(t - 1690) / 322)) ** (-1 / 0.39), t from the
# first control step. The training segment is her window t in [900, 3500] s
# (u = t - 900, u0 = 1690 - 900 = 790) renormalised to EXACT endpoints 270 / 235,
# so the match is bounded by what her curve still has to go at t = 3500 (0.32 deg).
EMILY_T0, EMILY_TAU, EMILY_NU = 1690.0, 322.0, 0.39
EMILY_WINDOW = (900.0, 3500.0)
RICH_T_TR = EMILY_WINDOW[1] - EMILY_WINDOW[0]      # 2600 s
RICH_T_HOLD = 1225.0
RICH_PERIOD = 2.0 * (RICH_T_HOLD + RICH_T_TR)      # 7650 s
RICH_NAMES = ("richards_270_235_phase", "richards_270_235_dr_phase",
              "mix_cycle_richards_phase", "cycle_stair_phase")
RICH_ABS_NAMES = RICH_NAMES
STAIR_LEVELS = (270.0, 258.0, 246.0, 235.0, 246.0, 258.0)
STAIR_PERIOD = len(STAIR_LEVELS) * (700.0 + 340.0)   # 6240 s
MTM30_SLEW = 0.1407     # les_3x3 frame slew cap at max_turb_move 30, dt_sim 5 (deg/s)
EPISODE_10000 = np.arange(0.0, 10000.0 + DT_SIM, DT_SIM)   # the exp_14/15 episode (1000 x dt_env 10)


def _emily(t):
    t = np.asarray(t, dtype=float)
    with np.errstate(over="ignore"):
        s = (1.0 + np.exp(-(t - EMILY_T0) / EMILY_TAU)) ** (-1.0 / EMILY_NU)
    return 270.0 + (235.0 - 270.0) * s


def _draw_episode_with_phase_zero(fn):
    """Trigger the t == 0 redraw, then pin the phase to 0 so the episode clock equals
    the cycle clock (t_episode = t_emily - 900 on the first transition)."""
    fn(0.0)
    fn.phase = 0.0


def test_stage15_names_are_registered_as_train_schedules_with_the_right_family():
    for name in RICH_NAMES + ("mix_static_richards",):
        assert name in TRAIN_WD_FACTORIES, name
        assert name not in WD_FUNCTIONS, name
        assert callable(get_train_wd_factory(name, seed=0))
    for name in RICH_ABS_NAMES:
        assert name in ABSOLUTE_TRAIN_NAMES, name
    assert "mix_static_richards" not in ABSOLUTE_TRAIN_NAMES


@pytest.mark.parametrize("name", RICH_ABS_NAMES)
def test_stage15_absolute_schedules_return_exactly_270_at_t0(name):
    """The trainer compares f(0) with the preset's pinned wd using ==, not approx
    (transformer_sac_windfarm.py pin-270 check): the renormalised segment must hand
    the hold value through bit-exactly on every episode."""
    from WindGym.core.wind_manager import WindManager
    fn = get_train_wd_factory(name, seed=5)
    assert fn.takes_base_wd is False
    assert WindManager._wd_function_takes_base_wd(fn) is False
    for _ in range(50):
        assert float(fn(0.0)) == 270.0


def test_make_richards_segment_hits_both_endpoints_exactly_and_is_monotone():
    g = make_richards_segment(270.0, 235.0, EMILY_TAU, EMILY_NU, RICH_T_TR)
    assert float(g(0.0)) == 270.0
    assert float(g(RICH_T_TR)) == 235.0
    u = np.arange(0.0, RICH_T_TR + 1.0, 1.0)
    wd = np.asarray(g(u), dtype=float)
    assert np.all(np.diff(wd) <= 0.0)
    assert g.T_tr == pytest.approx(RICH_T_TR)
    # reversed endpoints: the same shape going back up
    h = make_richards_segment(235.0, 270.0, EMILY_TAU, EMILY_NU, RICH_T_TR)
    assert float(h(0.0)) == 235.0 and float(h(RICH_T_TR)) == 270.0
    np.testing.assert_allclose(np.asarray(h(u)), 505.0 - wd)


def test_richards_270_235_phase_reproduces_emilys_profile_on_her_window():
    """phase = 0: the first 270 hold lasts 1225 s, then the transition is Emily's
    curve with t_emily = t_episode - 1225 + 900 = t_episode - 325. Max deviation on
    her window = the 0.32 deg her curve still has to go at t = 3500 (renormalised
    away); mean deviation well under 0.1 deg."""
    fn = get_train_wd_factory("richards_270_235_phase", seed=1)
    assert fn.period == pytest.approx(RICH_PERIOD)
    _draw_episode_with_phase_zero(fn)
    t_emily = np.arange(EMILY_WINDOW[0], EMILY_WINDOW[1] + 1.0, 1.0)
    t_episode = t_emily - EMILY_WINDOW[0] + RICH_T_HOLD
    wd = np.array([float(fn(t)) for t in t_episode])
    signed = wd - _emily(t_emily)
    dev = np.abs(signed)
    assert dev.max() < 0.35
    # the residual is the renormalisation of her tails: +0.05 deg at 900 s (her curve
    # has already left 270), falling monotonically to -0.32 deg at 3500 s (she is still
    # above 235); < 0.15 deg up to her peak-rate point (t_emily ~ 1990 s)
    assert np.all(np.diff(signed) <= 1e-9)
    assert dev[t_emily <= 1990.0].max() < 0.15
    # the segment is bracketed by exact holds and the period closes back at 270
    assert float(fn(RICH_T_HOLD - 1.0)) == 270.0
    assert float(fn(RICH_T_HOLD + RICH_T_TR + 1.0)) == 235.0
    assert float(fn(2.0 * RICH_T_HOLD + RICH_T_TR - 1.0)) == 235.0
    assert float(fn(RICH_PERIOD)) == 270.0
    # the return transition is the mirror image of the first one
    u = np.arange(0.0, RICH_T_TR + 1.0, 10.0)
    down = np.array([float(fn(RICH_T_HOLD + x)) for x in u])
    up = np.array([float(fn(2.0 * RICH_T_HOLD + RICH_T_TR + x)) for x in u])
    np.testing.assert_allclose(up, 505.0 - down, atol=1e-9)


def test_richards_270_235_phase_starts_inside_the_270_hold_and_phases_vary():
    fn = get_train_wd_factory("richards_270_235_phase", seed=5)
    phases = []
    for _ in range(100):
        wd = _trajectory(fn, grid=EPISODE_10000)
        assert wd[0] == 270.0
        assert 0.0 <= fn.phase < RICH_T_HOLD
        phases.append(fn.phase)
        first_move = EPISODE_10000[np.nonzero(np.abs(wd - 270.0) > 1e-9)[0][0]]
        assert first_move <= RICH_T_HOLD + DT_SIM
        assert wd.min() >= 235.0 - 1e-9 and wd.max() <= 270.0 + 1e-9
        assert np.isclose(wd, 235.0).any()          # the 235 hold is reached in a 10000 s episode
    phases = np.asarray(phases)
    assert phases.std() > 120.0
    assert len(np.unique(np.round(phases, 3))) > 90


def test_richards_dr_phase_redraws_tau_nu_and_hold_per_episode_inside_their_bands():
    fn = get_train_wd_factory("richards_270_235_dr_phase", seed=3)
    taus, nus, holds = [], [], []
    for _ in range(200):
        wd = _trajectory(fn, grid=EPISODE_10000)
        assert wd[0] == 270.0
        assert 160.0 <= fn.tau <= 640.0 and 0.25 <= fn.nu <= 1.0 and 800.0 <= fn.t_hold <= 1600.0
        assert 0.0 <= fn.phase < fn.t_hold
        assert fn.T_tr == pytest.approx(max(RICH_T_TR * fn.tau / EMILY_TAU, 400.0))
        assert fn.period == pytest.approx(2.0 * (fn.t_hold + fn.T_tr))
        assert wd.min() >= 235.0 - 1e-9 and wd.max() <= 270.0 + 1e-9
        rates = np.abs(np.diff(wd)) / DT_SIM
        assert rates.max() < MTM30_SLEW
        taus.append(fn.tau); nus.append(fn.nu); holds.append(fn.t_hold)
    assert np.std(taus) > 80.0 and np.std(nus) > 0.1 and np.std(holds) > 120.0


def test_mix_cycle_richards_phase_uses_both_branches_and_each_is_the_named_schedule():
    fn = get_train_wd_factory("mix_cycle_richards_phase", seed=7)
    branches = {"cycle": 0, "richards": 0}
    for _ in range(200):
        wd = _trajectory(fn, grid=EPISODE_10000)
        assert wd[0] == 270.0
        branches[fn.branch] += 1
        rates = np.abs(np.diff(wd)) / DT_SIM
        if fn.branch == "cycle":
            assert rates.max() == pytest.approx(SLOW_RATE)      # the R3 320 s linear ramps
        else:
            assert rates.max() < 0.074 and rates.max() >= 0.014
        assert wd.min() >= 235.0 - 1e-9 and wd.max() <= 270.0 + 1e-9
    assert branches["cycle"] >= 60 and branches["richards"] >= 60


def test_cycle_stair_phase_dwells_at_258_and_246_on_both_sides_of_the_crossing():
    fn = get_train_wd_factory("cycle_stair_phase", seed=2)
    assert fn.period == pytest.approx(STAIR_PERIOD)
    _draw_episode_with_phase_zero(fn)
    t = np.arange(1.0, STAIR_PERIOD + 1.0, 1.0)       # never t == 0 again: that would redraw the phase
    wd = np.array([float(fn(x)) for x in t])
    for level in (258.0, 246.0):
        assert np.mean(np.isclose(wd, level)) == pytest.approx(2 * 700.0 / STAIR_PERIOD, abs=0.002)
    assert np.mean(np.isclose(wd, 235.0)) == pytest.approx(700.0 / STAIR_PERIOD, abs=0.002)
    assert np.mean(np.isclose(wd, 270.0)) == pytest.approx(700.0 / STAIR_PERIOD, abs=0.002)
    rates = np.abs(np.diff(wd))
    assert set(np.round(rates[rates > 0], 6)) == {round(12.0 / 340.0, 6), round(11.0 / 340.0, 6)}
    # phase rule: only the 270 hold ([0, 700)) seeds the episode
    for _ in range(50):
        wd = _trajectory(fn)
        assert wd[0] == 270.0 and 0.0 <= fn.phase < 700.0


def test_mix_static_richards_is_relative_half_static_and_walks_inside_235_270():
    fn = get_train_wd_factory("mix_static_richards", seed=7)
    assert _takes_base_wd(fn)
    n_static, n_walk = 0, 0
    for base in (235.0, 250.0, 270.0):
        for _ in range(100):
            tr = _trajectory(fn, base_wd=base, grid=EPISODE_10000)
            assert tr[0] == base
            if np.allclose(tr, base):
                n_static += 1
                assert fn.branch == "static"
            else:
                n_walk += 1
                assert fn.branch == "walk"
                assert 235.0 - 1e-9 <= tr.min() and tr.max() <= 270.0 + 1e-9
                rates = np.abs(np.diff(tr)) / DT_SIM
                assert rates.max() <= 0.074 and rates.max() >= 0.006
                # the walk is a sequence of holds and Richards excursions, never a linear ramp:
                # the rate during a move is not constant
                moving = rates[rates > 1e-6]
                assert moving.std() > 1e-4
    assert 0.4 <= n_static / 300 <= 0.6
    assert n_walk > 0
