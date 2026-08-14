"""Named time-varying wind-direction schedules (change_wd / change_wd_2 sweeps).

Two registries live here and are deliberately kept disjoint:

``WD_FUNCTIONS`` -- deterministic EVAL schedules, ``wd_function(t) -> wd``.
    Returns the ABSOLUTE wind direction (deg) at simulation time ``t`` (seconds)
    and is passed straight to the pywake adapter via WindFarmEnv's
    ``wd_function`` kwarg. The env's burn-in holds the per-reset ``base_wd`` (see
    wind_manager.py), so any config using one of these must pin
    ``wd_min = wd_max = wd_function(0)`` to avoid a wind-direction jump at t=0
    (pattern: make_flow_gif.py).

``TRAIN_WD_FACTORIES`` -- randomized TRAINING schedules, ``wd_function(t, base_wd) -> wd``.
    RELATIVE, not absolute: ``wd(0) == base_wd``, so the schedule *composes* with
    the env's per-episode wd domain randomization (hard_2: wd~U[225,315]) rather
    than replacing it. That is why training arms must NOT pin wd_min/wd_max the
    way the eval path does. They also re-draw their episode parameters whenever
    called at ``t == 0.0``: ``make_wind_direction_list`` rebuilds the whole wd
    list once per reset, always sweeping t upward from 0, so that edge is a
    reliable one-per-episode hook.

Keeping the registries separate is a leakage guard -- an eval schedule name can
never be selected for training and vice versa (asserted in
tests/test_wd_functions.py, which also checks that the training family never
reproduces the ``step_ramp_270_315`` eval instance).

Registered by name so SLURM scripts can select a schedule with a plain string
(``--eval_wd_function step_ramp_270_315``, ``--train_wd_function dr_ramp``)
instead of importing code.
"""
from __future__ import annotations

from functools import partial

import numpy as np


def step_ramp_270_315(t):
    """The ``test_winddir_func`` schedule from test_changing_WD.ipynb, verbatim.

    270 -> 296.5 -> 315 deg: 300 s holds at each level, 200 s linear ramps between
    them. Full schedule completes at t = 1000 s.
    """
    # Goes from 270 -> 270+26.5 -> 270+45. It stays at each point for 100 steps.
    # Also moves over there in a linear fashion, so it is not a step function.
    t = np.asarray(t)

    wd_1 = 270
    wd_2 = 270 + 26.5
    wd_3 = 270 + 45

    time_at_wd = 300  # Time spent at each wind direction
    time_to_change = 200  # Time taken to change from one wind direction to the next

    wd = np.zeros_like(t, dtype=float)
    wd[t < time_at_wd] = wd_1
    wd[(t >= time_at_wd) & (t < time_at_wd + time_to_change)] = wd_1 + (wd_2 - wd_1) * ((t[(t >= time_at_wd) & (t < time_at_wd + time_to_change)] - time_at_wd) / time_to_change)
    wd[(t >= time_at_wd + time_to_change) & (t < 2 * time_at_wd + time_to_change)] = wd_2
    wd[(t >= 2 * time_at_wd + time_to_change) & (t < 2 * time_at_wd + 2 * time_to_change)] = wd_2 + (wd_3 - wd_2) * ((t[(t >= 2 * time_at_wd + time_to_change) & (t < 2 * time_at_wd + 2 * time_to_change)] - (2 * time_at_wd + time_to_change)) / time_to_change)
    wd[t >= 2 * time_at_wd + 2 * time_to_change] = wd_3

    return wd


def static_270(t):
    """Constant 270 deg — static-wd control matching step_ramp_270_315's start."""
    t = np.asarray(t)
    return np.full_like(t, 270.0, dtype=float)


def static_315(t):
    """Constant 315 deg — static-wd control matching step_ramp_270_315's END.

    The change_wd_2 verdict flagged the missing half of the static control: with
    only static_270 the ramp-vs-static gap conflates "tracking a moving wd is
    hard" with "315 deg simply has less to gain". On the square_5x5 @6D eval
    layout 270 deg is row-aligned (deep wakes) while 315 deg is the diagonal at
    8.49D (shallow wakes), so the two endpoints are not interchangeable and the
    retention ratio is uninterpretable without both.
    """
    t = np.asarray(t)
    return np.full_like(t, 315.0, dtype=float)


def hold_ramp_270_315(t):
    """Two steady states with one slow transition between them (Marcus).

    270 deg for 600 s, a single linear 45 deg ramp over 1200 s
    (0.0375 deg/s), then 315 deg for 600 s. Episode completes at t = 2400 s.

    Why this exists alongside ``step_ramp_270_315``: that schedule packs TWO
    ramps (270 -> 296.5 -> 315) with 300 s holds and 200 s transitions, i.e.
    0.09-0.13 deg/s, which is 5-7x the wd_slow frame's rate limit and leaves
    the policy no settled interval to be read off. This one isolates
    steady -> transition -> steady.

    NOTE the transition is still ~2x the frame's rate limit (0.0182 deg/s on
    square_3x3 at dt_sim=5, = max_turb_move*360/(2*pi*max_dist)/dt_sim), so
    ``env.wd`` will not track it exactly -- MetmastSite's wd_slow is the mean
    of a FORWARD and a BACKWARD rate-limited pass, so it leads the ramp before
    it starts and trails it after (see HANDOVER section 4 finding 1). The
    residual goes into wd_small as inflow tilt, so the flow the rotors
    actually see is always this schedule exactly. A ramp slower than
    0.0182 deg/s (>= ~2475 s for 45 deg) would make wd_slow == this schedule
    with wd_small == 0.
    """
    t = np.asarray(t)

    wd_lo, wd_hi = 270.0, 315.0
    t_hold = 600.0          # seconds at each steady direction
    t_ramp = 1200.0         # seconds of transition -> 0.0375 deg/s

    wd = np.full_like(t, wd_lo, dtype=float)
    ramping = (t >= t_hold) & (t < t_hold + t_ramp)
    wd[ramping] = wd_lo + (wd_hi - wd_lo) * ((t[ramping] - t_hold) / t_ramp)
    wd[t >= t_hold + t_ramp] = wd_hi
    return wd


WD_FUNCTIONS = {
    "step_ramp_270_315": step_ramp_270_315,
    "hold_ramp_270_315": hold_ramp_270_315,
    "static_270": static_270,
    "static_315": static_315,
}


def parse_eval_ws(eval_ws) -> list:
    """Parse ``--eval_ws`` ("12" or "12,10.5") into a list of floats.

    Order is preserved: the FIRST entry pairs with the first schedule to form the
    spec that owns the unprefixed ``eval/...`` keys.
    """
    if eval_ws is None:
        return [12.0]
    if isinstance(eval_ws, (int, float)):
        return [float(eval_ws)]
    parts = [s.strip() for s in str(eval_ws).split(",") if s.strip()]
    if not parts:
        return [12.0]
    return [float(p) for p in parts]


def build_eval_specs(eval_wd_function, eval_ws="12") -> list:
    """Cross (wd schedule x eval ws) into the ordered eval ladder.

    Returns ``[(wd_name, ws, spec_name), ...]``, empty when no --eval_wd_function
    was given (the caller then falls back to the single static-wd evaluator).

    ``spec_name`` is the W&B namespace segment under ``eval/wd/``. With exactly one
    requested wind speed it is the bare schedule name, making the key set
    byte-identical to change_wd_2's; only when several speeds are requested does
    the ``/ws<speed>`` segment appear. That asymmetry is deliberate -- it is the
    one change here that could silently invalidate a cross-sweep comparison.

    The first spec is (first schedule, first speed), and the caller additionally
    writes the historical unprefixed ``eval/...`` keys from it.
    """
    if eval_wd_function is None or not str(eval_wd_function).strip():
        return []
    names = [s.strip() for s in str(eval_wd_function).split(",") if s.strip()]
    if not names:
        return []
    for name in names:
        get_wd_function(name)  # fail fast on a typo, before any env is built
    ws_list = parse_eval_ws(eval_ws)
    multi_ws = len(ws_list) > 1
    return [
        (wd, ws, f"{wd}/ws{ws:g}" if multi_ws else wd)
        for wd in names
        for ws in ws_list
    ]


def get_wd_function(name: str):
    """Look up a registered EVAL wd_function by name (ValueError lists known names)."""
    try:
        return WD_FUNCTIONS[name]
    except KeyError:
        raise ValueError(
            f"unknown wd_function '{name}' (known: {sorted(WD_FUNCTIONS)})"
        ) from None


# ---------------------------------------------------------------------------
# Randomized TRAINING schedules
# ---------------------------------------------------------------------------

class _DrRampSchedule:
    """A per-episode piecewise-linear hold/ramp walk in wind direction.

    Callable as ``schedule(t, base_wd) -> wd``. Episode parameters (segment
    count, hold durations, ramp rates, signed excursions) are re-drawn on every
    call with ``t == 0.0``; between re-draws the walk is a pure lookup, so
    sampling the same episode twice is deterministic.

    The walk is stored as *deltas* from ``base_wd`` with ``delta(0) = 0``, which
    is what keeps it composable with the env's wd domain randomization: there is
    no discontinuity where the burn-in (which holds ``base_wd``) hands over to
    the episode.

    Excursions that would leave ``[wd_lo, wd_hi]`` are REFLECTED (sign flipped)
    rather than clipped. Reflection preserves the drawn ramp rate exactly, so
    the declared rate band is a hard guarantee; clipping would instead
    manufacture spurious near-zero rates at the domain edges.
    """

    def __init__(
        self,
        seed: int,
        rate_lo: float,
        rate_hi: float,
        exc_lo: float,
        exc_hi: float,
        wd_lo: float = 225.0,
        wd_hi: float = 315.0,
        hold_lo: float = 100.0,
        hold_hi: float = 600.0,
        t_horizon: float = 20000.0,
    ):
        if not (rate_lo > 0 and rate_hi >= rate_lo):
            raise ValueError(f"bad rate band [{rate_lo}, {rate_hi}]")
        if not (exc_lo > 0 and exc_hi >= exc_lo):
            raise ValueError(f"bad excursion band [{exc_lo}, {exc_hi}]")
        if not wd_hi > wd_lo:
            raise ValueError(f"bad wd domain [{wd_lo}, {wd_hi}]")

        self._rng = np.random.default_rng(seed)
        self._rate_lo, self._rate_hi = float(rate_lo), float(rate_hi)
        self._exc_lo, self._exc_hi = float(exc_lo), float(exc_hi)
        self._wd_lo, self._wd_hi = float(wd_lo), float(wd_hi)
        self._hold_lo, self._hold_hi = float(hold_lo), float(hold_hi)
        self._t_horizon = float(t_horizon)

        # Upper bound on the segment pairs needed to span the horizon, so the
        # whole episode's randomness can be drawn in four vectorized calls.
        min_pair = self._hold_lo + self._exc_lo / self._rate_hi
        self._n_pairs = int(np.ceil(self._t_horizon / min_pair)) + 2

        self._knot_t = np.zeros(1)
        self._knot_d = np.zeros(1)

    def _draw(self, base_wd: float) -> None:
        rng = self._rng
        n = self._n_pairs
        holds = rng.uniform(self._hold_lo, self._hold_hi, size=n)
        mags = rng.uniform(self._exc_lo, self._exc_hi, size=n)
        rates = rng.uniform(self._rate_lo, self._rate_hi, size=n)
        signs = rng.choice(np.array([-1.0, 1.0]), size=n)

        ts = [0.0]
        ds = [0.0]
        t = 0.0
        d = 0.0
        for i in range(n):
            # Hold at the current direction.
            t += float(holds[i])
            ts.append(t)
            ds.append(d)

            # Ramp to a new direction, kept inside the wd domain.
            mag = float(mags[i])
            sign = float(signs[i])
            here = base_wd + d
            if not (self._wd_lo <= here + sign * mag <= self._wd_hi):
                sign = -sign
            if not (self._wd_lo <= here + sign * mag <= self._wd_hi):
                # Excursion wider than the domain itself: take the longer side.
                room_up = self._wd_hi - here
                room_dn = here - self._wd_lo
                sign = 1.0 if room_up >= room_dn else -1.0
                mag = max(room_up, room_dn)
            if mag <= 0.0:
                continue  # pinned against a degenerate domain; just keep holding

            t += mag / float(rates[i])
            d += sign * mag
            ts.append(t)
            ds.append(d)

            if t >= self._t_horizon:
                break

        self._knot_t = np.asarray(ts, dtype=float)
        self._knot_d = np.asarray(ds, dtype=float)

    def __call__(self, t, base_wd):
        t = float(t)
        if t == 0.0:
            self._draw(float(base_wd))
        return float(base_wd) + float(np.interp(t, self._knot_t, self._knot_d))


def make_dr_ramp(
    seed: int,
    rate_lo: float,
    rate_hi: float,
    exc_lo: float,
    exc_hi: float,
    wd_lo: float = 225.0,
    wd_hi: float = 315.0,
    **kwargs,
):
    """Build a seeded randomized training wd schedule, ``f(t, base_wd) -> wd``."""
    return _DrRampSchedule(
        seed=seed,
        rate_lo=rate_lo, rate_hi=rate_hi,
        exc_lo=exc_lo, exc_hi=exc_hi,
        wd_lo=wd_lo, wd_hi=wd_hi,
        **kwargs,
    )


# dr_ramp: broad DR. The rate band brackets the eval schedule's 0.0925 and
# 0.1325 deg/s by roughly a factor of 2 on each side.
# dr_ramp_narrow: a *family* around the eval instance (45 deg total at ~0.1
# deg/s) -- never the instance itself; see the leakage test.
# dr_ramp_slow: QUASI-STATIC drift (Marcus 2026-08-13). rate_hi is chosen
# BELOW the wd_slow frame slew limit of the largest DR training farms
# (max_wd_step = max_turb_move*360/(2*pi*max_dist) per sim step ~= 0.019
# deg/s for 16-turbine grids at dt_sim=5), so on dynamiks the WHOLE
# excursion is tracked by the frame rotation: wd_small stays ~0, the wd
# change only moves the MEAN conditions, and no farm (baseline, exploring
# agent) can be pushed toward cut-in by an untrackable inflow tilt. A full
# 40-50 deg excursion takes 2700-10000 s, i.e. roughly one slow sweep per
# 5000 s training episode.
TRAIN_WD_FACTORIES = {
    "dr_ramp": partial(make_dr_ramp, rate_lo=0.02, rate_hi=0.25,
                       exc_lo=10.0, exc_hi=60.0),
    "dr_ramp_narrow": partial(make_dr_ramp, rate_lo=0.09, rate_hi=0.14,
                              exc_lo=40.0, exc_hi=50.0),
    "dr_ramp_slow": partial(make_dr_ramp, rate_lo=0.005, rate_hi=0.015,
                            exc_lo=40.0, exc_hi=50.0),
}


def get_train_wd_factory(name: str, seed: int):
    """Build a registered TRAINING wd schedule (ValueError lists known names).

    Each vector env must get its own seed, otherwise all 30 envs walk the same
    wd trajectory and the "train on changing wd" arms collapse to a single
    schedule per episode index.
    """
    try:
        factory = TRAIN_WD_FACTORIES[name]
    except KeyError:
        raise ValueError(
            f"unknown train_wd_function '{name}' "
            f"(known: {sorted(TRAIN_WD_FACTORIES)}); eval-only schedules "
            f"({sorted(WD_FUNCTIONS)}) may not be used for training"
        ) from None
    return factory(seed=seed)
