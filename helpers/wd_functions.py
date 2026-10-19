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

A THIRD family lives inside ``TRAIN_WD_FACTORIES`` but is ABSOLUTE
(LES-3x3 Stage 5, ``cycle_270_235_phase``): ``wd_function(t) -> wd`` like an
eval schedule, seeded, and re-drawing only a random START PHASE at ``t == 0.0``.
The shape (270 hold / ramp / 235 hold / ramp back) is fixed and the schedule
never composes with the env's wd band -- instead the preset must pin
``wd_min = wd_max = f(0)`` exactly as an eval schedule would (``les_recipe_pin270``),
and the trainer hard-errors otherwise. Names of this family are listed in
``ABSOLUTE_TRAIN_NAMES``; ``WindManager._wd_function_takes_base_wd`` sees the
one-argument ``__call__`` and treats them as absolute automatically.

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


def static_235(t):
    """Constant 235 deg — static-wd control matching hold_ramp_270_235's END.

    Same rationale as static_315: without the endpoint control, the
    ramp-vs-static gap conflates "tracking a moving wd is hard" with "the
    endpoint simply has less to gain". On les_3x3 (7D x 5D grid) this is
    acute: atan2(891.5, 1248.1) = 35.5 deg puts 235 deg almost exactly on
    the shallow-wake diagonal (234.5 deg), while 270 deg is deep-wake
    row-aligned — the two endpoints have very different wake-steering
    headroom, so the scenario metric must be whole-trajectory energy, not
    an end-state ratio.
    """
    t = np.asarray(t)
    return np.full_like(t, 235.0, dtype=float)


def make_hold_ramp(wd_from: float, wd_to: float, t_hold_pre: float, t_ramp: float):
    """Build a hold -> single linear ramp -> hold eval schedule, ``f(t) -> wd``.

    Generalizes ``hold_ramp_270_315`` (kept verbatim above for T3 provenance):
    ``wd_from`` until ``t_hold_pre``, then a linear ramp to ``wd_to`` over
    ``t_ramp`` seconds, then ``wd_to`` forever. The trailing hold has no end —
    the episode length decides how much of it is sampled.

    Registered instances (LES-3x3 campaign): holds are sized as 6 flow
    passthroughs of the les_3x3 farm (max pairwise extent 3067.6 m) at the
    nominal 9 m/s -> 2045 s; the 35 deg / 600 s ramp is 0.0583 deg/s, which the
    mean-flow frame tracks EXACTLY once max_turb_move raises the frame slew
    limit above it (mtm=12 -> 0.0897 deg/s on les_3x3), so wd_small stays 0 and
    env.wd is exact and causal — unlike hold_ramp_270_315 at the 2 m default.
    """
    wd_from, wd_to = float(wd_from), float(wd_to)
    t_hold_pre, t_ramp = float(t_hold_pre), float(t_ramp)
    if t_ramp <= 0:
        raise ValueError(f"t_ramp must be > 0, got {t_ramp}")

    def _hold_ramp(t):
        t = np.asarray(t)
        wd = np.full_like(t, wd_from, dtype=float)
        ramping = (t >= t_hold_pre) & (t < t_hold_pre + t_ramp)
        wd[ramping] = wd_from + (wd_to - wd_from) * ((t[ramping] - t_hold_pre) / t_ramp)
        wd[t >= t_hold_pre + t_ramp] = wd_to
        return wd

    _hold_ramp.__name__ = f"hold_ramp_{wd_from:g}_{wd_to:g}"
    _hold_ramp.__doc__ = (
        f"{wd_from:g} deg for {t_hold_pre:g} s, linear ramp to {wd_to:g} deg over "
        f"{t_ramp:g} s ({abs(wd_to - wd_from) / t_ramp:.4f} deg/s), then hold."
    )
    return _hold_ramp


def make_static(wd: float):
    """Build a constant-wd eval schedule ``f(t) -> wd`` (see static_270/235).

    Registered instances (LES-3x3 Stage 3): the 265/275 endpoints of the
    narrow-band retreat (les_recipe_nb, wd ~ U[265, 275]).
    """
    wd = float(wd)

    def _static(t):
        t = np.asarray(t)
        return np.full_like(t, wd, dtype=float)

    _static.__name__ = f"static_{wd:g}"
    _static.__doc__ = f"Constant {wd:g} deg."
    return _static


def make_cycle(wd_a: float, wd_b: float, t_hold: float, t_ramp: float, phase: float = 0.0):
    """Build a periodic hold/ramp/hold/ramp-back eval schedule ``f(t) -> wd``.

    One period (``2 * (t_hold + t_ramp)`` s) is: hold ``wd_a`` for ``t_hold``,
    linear ramp to ``wd_b`` over ``t_ramp``, hold ``wd_b`` for ``t_hold``, linear
    ramp back to ``wd_a`` over ``t_ramp``; then repeat. ``tau = (t + phase) %
    period`` selects the segment, so ``phase`` shifts where t=0 lands in the
    cycle (``f(0) == wd_a`` for any ``0 <= phase < t_hold``).

    LES-3x3 Stage 5 trains ON the 270 <-> 235 transition: 1000 s holds and 200 s
    ramps (0.175 deg/s) -- the ramp is faster than the Stage-2/3 scenario ramp
    (600 s, 0.0583 deg/s) and needs ``max_turb_move >= 23.4`` for the mean-flow
    frame to track it (campaign choice mtm=25 -> 0.1868 deg/s slew limit).
    """
    wd_a, wd_b = float(wd_a), float(wd_b)
    t_hold, t_ramp, phase = float(t_hold), float(t_ramp), float(phase)
    if t_ramp <= 0 or t_hold < 0:
        raise ValueError(f"need t_ramp > 0 and t_hold >= 0, got {t_ramp}, {t_hold}")
    period = 2.0 * (t_hold + t_ramp)

    def _cycle(t):
        t = np.asarray(t, dtype=float)
        tau = np.mod(t + phase, period)
        wd = np.full_like(t, wd_a, dtype=float)
        # ramp a -> b
        m = (tau >= t_hold) & (tau < t_hold + t_ramp)
        wd[m] = wd_a + (wd_b - wd_a) * ((tau[m] - t_hold) / t_ramp)
        # hold b
        m = (tau >= t_hold + t_ramp) & (tau < 2.0 * t_hold + t_ramp)
        wd[m] = wd_b
        # ramp b -> a
        m = tau >= 2.0 * t_hold + t_ramp
        wd[m] = wd_b + (wd_a - wd_b) * ((tau[m] - (2.0 * t_hold + t_ramp)) / t_ramp)
        return wd

    _cycle.__name__ = f"cycle_{wd_a:g}_{wd_b:g}"
    _cycle.__doc__ = (
        f"Periodic {wd_a:g} <-> {wd_b:g} deg: {t_hold:g} s holds, {t_ramp:g} s ramps "
        f"({abs(wd_b - wd_a) / t_ramp:.4f} deg/s), period {period:g} s, phase {phase:g} s."
    )
    _cycle.period = period
    return _cycle


def make_cycle_knots(levels, t_hold: float, t_ramp: float, phase: float = 0.0):
    """Build a periodic multi-level hold/ramp eval schedule ``f(t) -> wd`` from knots.

    ``levels = (L0, L1, ..., Ln-1)``: hold ``L0`` for ``t_hold``, ramp linearly to
    ``L1`` over ``t_ramp``, hold ``L1``, ... ramp from ``Ln-1`` back to ``L0``; then
    repeat (period ``n * (t_hold + t_ramp)``). Evaluated with ``np.interp`` over
    the knots, so two levels reproduce ``make_cycle`` exactly. ``phase`` shifts
    where t=0 lands (``f(0) == L0`` for ``0 <= phase < t_hold``).

    LES-3x3 Stage 12 (exp_12 `sym`): ``(270, 235, 270, 305)`` at the _slow timing
    (1600 s holds, 320 s ramps, 0.1094 deg/s) is the two-sided training cycle,
    period 7680 s, 42 % of the time at exactly 270.
    """
    levels = tuple(float(v) for v in levels)
    t_hold, t_ramp, phase = float(t_hold), float(t_ramp), float(phase)
    if len(levels) < 2:
        raise ValueError(f"need at least two levels, got {levels}")
    if t_ramp <= 0 or t_hold < 0:
        raise ValueError(f"need t_ramp > 0 and t_hold >= 0, got {t_ramp}, {t_hold}")
    seg = t_hold + t_ramp
    period = len(levels) * seg
    knots_t, knots_v = [], []
    for k, v in enumerate(levels):
        knots_t += [k * seg, k * seg + t_hold]
        knots_v += [v, v]
    knots_t.append(period)
    knots_v.append(levels[0])
    knots_t = np.asarray(knots_t, dtype=float)
    knots_v = np.asarray(knots_v, dtype=float)

    def _cycle(t):
        t = np.asarray(t, dtype=float)
        tau = np.mod(t + phase, period)
        return np.interp(tau, knots_t, knots_v)

    _cycle.__name__ = "cycle_" + "_".join(f"{v:g}" for v in levels)
    _cycle.__doc__ = (
        f"Periodic {' -> '.join(f'{v:g}' for v in levels)} -> {levels[0]:g} deg: {t_hold:g} s holds, "
        f"{t_ramp:g} s ramps, period {period:g} s, phase {phase:g} s."
    )
    _cycle.period = period
    _cycle.levels = levels
    return _cycle


WD_FUNCTIONS = {
    "step_ramp_270_315": step_ramp_270_315,
    "hold_ramp_270_315": hold_ramp_270_315,
    "static_270": static_270,
    "static_315": static_315,
    "static_235": static_235,
    # --- LES-3x3 scenario family (les_3x3 layout, DTU10MW, see make_hold_ramp) ---
    # The target scenario: 6 passthroughs @270 (deep-wake rows), 35 deg down-ramp
    # in 600 s, then 235 (shallow-wake diagonal). 470 env steps @ dt_env=10.
    "hold_ramp_270_235": make_hold_ramp(270.0, 235.0, 2045.0, 600.0),
    # Short-hold variant that fits the 140-step in-training eval window.
    "hold_ramp_270_235_short": make_hold_ramp(270.0, 235.0, 300.0, 600.0),
    # Rate-matched VEER mirror (270 -> +35 deg) for the up/down asymmetry check.
    "hold_ramp_270_305": make_hold_ramp(270.0, 305.0, 2045.0, 600.0),
    # --- LES-3x3 Stage 3: narrow-band retreat (les_recipe_nb, wd in [265, 275]) ---
    # In-band statics at the band edges (270 is the centre, kept above).
    "static_265": make_static(265.0),
    "static_275": make_static(275.0),
    # 5 deg in-band ramps at 0.0167 deg/s (300 s = 30 env steps @ dt_env=10):
    # slow enough that the ramp is windowable; the scenario rate (0.0583 deg/s)
    # would make a 5 deg ramp only ~9 steps. Same 2045 s pre-hold -> 470 steps.
    "hold_ramp_270_275": make_hold_ramp(270.0, 275.0, 2045.0, 300.0),
    "hold_ramp_270_265": make_hold_ramp(270.0, 265.0, 2045.0, 300.0),
    # Short-hold variant for the 140-step in-training eval (300 s hold + 300 s ramp).
    "hold_ramp_270_275_short": make_hold_ramp(270.0, 275.0, 300.0, 300.0),
    # --- LES-3x3 Stage 5: train ON the transition (cyclic 270 <-> 235) ---
    # Deterministic eval instance of the training cycle (phase 0): 1000 s holds,
    # 200 s ramps (0.175 deg/s; needs max_turb_move >= 23.4 -> campaign mtm=25),
    # period 2400 s. Two periods = 4800 s = 480 env steps @ dt_env=10 (harvest
    # uses 490 steps, which is < the ws-10 eval env time_max of 6136 s).
    "cycle_270_235": make_cycle(270.0, 235.0, 1000.0, 200.0),
    # 500 s holds -> period 1400 s = exactly the 140-step in-training eval window.
    "cycle_270_235_short": make_cycle(270.0, 235.0, 500.0, 200.0),
    # --- LES-3x3 Stage 6: IEA 22 MW "real" case (_slow family) ---
    # The Stage-5 cycle scaled 1.6x in TIME (advection-equivalent at the IEA22
    # D=284 m: D-ratio 284/178.3 = 1.593): 1600 s holds, 320 s ramps
    # (0.1094 deg/s), period 3840 s. The suffix is RATE-descriptive, not
    # turbine-specific -- any farm whose frame slew limit exceeds 0.109 deg/s
    # can track it (IEA22 les_3x3 needs max_turb_move >= 23.4; campaign mtm=30
    # -> 0.1407 deg/s, 29 % margin). Two periods = 768 env steps @ dt_env=10.
    "cycle_270_235_slow": make_cycle(270.0, 235.0, 1600.0, 320.0),
    # 800 s holds -> period 2240 s = exactly the 224-step in-training eval
    # window (--num_eval_steps 224).
    "cycle_270_235_slow_short": make_cycle(270.0, 235.0, 800.0, 320.0),
    # --- LES-3x3 Stage 12 (exp_12 `sym`): two-sided _slow cycle ---
    # Deterministic eval twin of cycle_270_235_305_slow_phase (phase 0): 270 ->
    # 235 -> 270 -> 305 -> 270, 1600 s holds, 320 s ramps (0.1094 deg/s), period
    # 7680 s; wd is at exactly 270 for 42 % of the period and above / below it
    # for equal shares, so the yaw sign is defined on BOTH sides of 270.
    "cycle_270_235_305_slow": make_cycle_knots((270.0, 235.0, 270.0, 305.0), 1600.0, 320.0),
    # Rate-transfer hold/ramp at the _slow ramp rate: 6 flow passthroughs of
    # the IEA22 les_3x3 max pairwise extent (4886.1 m) at the nominal 9 m/s
    # -> 3257 s pre-hold, then the 35 deg / 320 s ramp. 470-step cell like
    # hold_ramp_270_235.
    "hold_ramp_270_235_slow": make_hold_ramp(270.0, 235.0, 3257.0, 320.0),
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

    def _draw_ramp_shapes(self, n: int):
        """Per-ramp shape draw (hook): the linear walk draws a rate per ramp."""
        return self._rng.uniform(self._rate_lo, self._rate_hi, size=n)

    def _ramp_duration(self, shape, mag: float) -> float:
        """Seconds a ramp of ``mag`` deg takes for the drawn ``shape`` (hook)."""
        return mag / float(shape)

    def _draw(self, base_wd: float) -> None:
        rng = self._rng
        n = self._n_pairs
        holds = rng.uniform(self._hold_lo, self._hold_hi, size=n)
        mags = rng.uniform(self._exc_lo, self._exc_hi, size=n)
        shapes = self._draw_ramp_shapes(n)
        signs = rng.choice(np.array([-1.0, 1.0]), size=n)

        ts = [0.0]
        ds = [0.0]
        ramps = []   # (ramp index i, knot index of the ramp start) for every realised ramp
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

            ramps.append((i, len(ts) - 1))
            t += self._ramp_duration(shapes[i], mag)
            d += sign * mag
            ts.append(t)
            ds.append(d)

            if t >= self._t_horizon:
                break

        self._knot_t = np.asarray(ts, dtype=float)
        self._knot_d = np.asarray(ds, dtype=float)
        self._ramps = ramps
        self._shapes = shapes

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


class _CycleTrainSchedule:
    """ABSOLUTE cyclic training schedule with a per-episode random start phase.

    Callable as ``schedule(t) -> wd`` -- ONE positional argument, so
    ``WindManager._wd_function_takes_base_wd`` classifies it as absolute and the
    env never hands it ``base_wd``. ``takes_base_wd = False`` documents this.

    At every call with ``t == 0.0`` (once per reset, see module docstring) a new
    ``phase ~ U[0, phase_max)`` is drawn; the episode then returns
    ``make_cycle(...)(t + phase)``. With ``phase_max <= t_hold`` the episode
    always starts in the ``wd_a`` hold, so ``f(0) == wd_a`` and the burn-in
    (which holds the preset's pinned wd) hands over without a jump.
    """

    takes_base_wd = False

    def __init__(self, seed: int, wd_a: float, wd_b: float, t_hold: float,
                 t_ramp: float, phase_max: float, levels=None):
        phase_max = float(phase_max)
        if not (0.0 <= phase_max <= float(t_hold)):
            raise ValueError(
                f"phase_max must lie in [0, t_hold={t_hold}] so that f(0) == wd_a "
                f"(no jump at the burn-in boundary); got {phase_max}"
            )
        self._rng = np.random.default_rng(seed)
        self._wd_a, self._wd_b = float(wd_a), float(wd_b)
        self._t_hold, self._t_ramp = float(t_hold), float(t_ramp)
        self._phase_max = phase_max
        if levels is None:
            self._base = make_cycle(wd_a, wd_b, t_hold, t_ramp)
            self._hold_starts = [0.0]
        else:
            # Stage 12 multi-level cycle: the phase is drawn inside ANY hold of the
            # first level (for (270, 235, 270, 305): [0, t_hold) or [2 seg, 2 seg +
            # t_hold), equal probability), so f(0) == levels[0] always and either
            # excursion can come first.
            self._base = make_cycle_knots(levels, t_hold, t_ramp)
            seg = float(t_hold) + float(t_ramp)
            self._hold_starts = [k * seg for k, v in enumerate(self._base.levels)
                                 if v == self._base.levels[0]]
        self.period = self._base.period
        self.phase = 0.0

    def _draw(self) -> None:
        start = self._hold_starts[0]
        if len(self._hold_starts) > 1:
            start = self._hold_starts[int(self._rng.integers(len(self._hold_starts)))]
        self.phase = start + (float(self._rng.uniform(0.0, self._phase_max)) if self._phase_max > 0 else 0.0)

    def __call__(self, t):
        t = float(t)
        if t == 0.0:
            self._draw()
        return float(self._base(t + self.phase))


def make_cycle_train(wd_a: float, wd_b: float, t_hold: float, t_ramp: float,
                     seed: int, phase_max: float, levels=None):
    """Build a seeded ABSOLUTE cyclic training schedule, ``f(t) -> wd``.

    ``levels`` (Stage 12) replaces the two-level ``wd_a <-> wd_b`` cycle by the
    knot cycle ``make_cycle_knots(levels, ...)``; ``wd_a`` / ``wd_b`` are then
    only informative (``levels[0]`` must equal ``wd_a``).
    """
    if levels is not None and float(levels[0]) != float(wd_a):
        raise ValueError(f"levels[0]={levels[0]} must equal wd_a={wd_a} (the pinned burn-in wd)")
    return _CycleTrainSchedule(seed=seed, wd_a=wd_a, wd_b=wd_b, t_hold=t_hold,
                               t_ramp=t_ramp, phase_max=phase_max, levels=levels)


class _MixStaticRamp:
    """Stage 14 ``mix_static_ramp``: per episode, with probability ``p_static``
    hold ``base_wd`` for the whole episode (the static-band control, so the
    policy sees the preset's U[wd_min, wd_max] directions as static states),
    otherwise run a ``_DrRampSchedule`` walk (hold / ramp transients at many
    directions). Relative (2-arg ``f(t, base_wd)``), re-draws the coin at
    ``t == 0.0``; the ramp's own ``t == 0`` redraw fires on the same call, so a
    static episode never advances the ramp's RNG and a ramp episode always
    starts from ``base_wd`` (``delta(0) = 0``).
    """

    def __init__(self, seed: int, p_static: float, **ramp_kwargs):
        if not 0.0 <= float(p_static) <= 1.0:
            raise ValueError(f"p_static must be in [0, 1], got {p_static}")
        self._rng = np.random.default_rng(seed)
        self._p = float(p_static)
        self._ramp = _DrRampSchedule(seed=int(self._rng.integers(2**31 - 1)), **ramp_kwargs)
        self._static = True

    def __call__(self, t, base_wd):
        t = float(t)
        if t == 0.0:
            self._static = bool(self._rng.random() < self._p)
        if self._static:
            return float(base_wd)
        return self._ramp(t, base_wd)


def make_mix_static_ramp(
    seed: int,
    p_static: float = 0.7,
    rate_lo: float = 0.09,
    rate_hi: float = 0.14,
    exc_lo: float = 30.0,
    exc_hi: float = 45.0,
    wd_lo: float = 225.0,
    wd_hi: float = 315.0,
    **kwargs,
):
    """Build the Stage-14 static / ramp mixture (see ``_MixStaticRamp``)."""
    return _MixStaticRamp(seed, p_static, rate_lo=rate_lo, rate_hi=rate_hi,
                          exc_lo=exc_lo, exc_hi=exc_hi, wd_lo=wd_lo, wd_hi=wd_hi,
                          **kwargs)


# ---------------------------------------------------------------------------
# Stage 15 (exp_15, cell R): Emily's generalized-logistic (Richards) wd turn
# ---------------------------------------------------------------------------
# Emily's LES profile (LESRL dwm_cli richards:270:235:1690:322:0.39):
#   wd(t) = 270 + (235 - 270) * (1 + exp(-(t - 1690) / 322)) ** (-1 / 0.39)
# (t from the first control step, 3600 s). Its slow turn peaks at 0.0336 deg/s
# and crosses the les_3x3 wake-alignment direction 250.3 deg over ~5 min; the
# Stage-14 arms (0.109-0.14 deg/s linear ramps) never dwelt there and lost
# 7-18 % to greedy on wd 265 -> 255 (cell R, 2026-10-19). The schedules below
# train on that shape. The training segment is Emily's window t in [900, 3500]
# s (|wd - 270| < 0.06 deg at 900 s; her curve is still 0.32 deg above 235 at
# 3500 s), renormalised so the segment starts and ends EXACTLY on the holds:
# the trainer's pin-270 check compares f(0) with the preset wd using ==.
RICHARDS_TAU = 322.0            # Emily's tau [s]
RICHARDS_NU = 0.39              # Emily's nu (asymmetry)
RICHARDS_T_TR = 2600.0          # her 900 -> 3500 s transition window [s]
RICHARDS_U0 = 1690.0 - 900.0    # her t0 inside that window [s]
RICHARDS_T_TR_MIN = 400.0       # floor on a DR-scaled transition [s]
RICHARDS_T_HOLD = 1225.0        # holds of the fixed cycle: period 7650 s
# DR bands (richards_270_235_dr_phase / mix_cycle_richards_phase / mix_static_richards):
# tau 160-640 s x nu 0.25-1.0 -> peak rates 0.0149-0.0727 deg/s for a 35 deg
# turn (all below the 0.1407 deg/s les_3x3 frame slew cap at max_turb_move 30).
RICHARDS_DR = {"tau": (160.0, 640.0), "nu": (0.25, 1.0), "t_hold": (800.0, 1600.0)}


def make_richards_segment(wd_a: float, wd_b: float, tau: float, nu: float, T_tr: float,
                          u0: float | None = None):
    """Richards transition ``g(u) -> wd`` on ``u in [0, T_tr]`` from ``wd_a`` to ``wd_b``.

    ``s(u) = (1 + exp(-(u - u0) / tau)) ** (-1 / nu)`` (Emily's formula), with
    ``u0 = T_tr * 790 / 2600`` by default so the shape sits in the segment where
    it sits in her 900-3500 s window for any ``T_tr = 2600 * tau / 322``, then
    renormalised ``(s(u) - s(0)) / (s(T_tr) - s(0))`` so ``g(0) == wd_a`` and
    ``g(T_tr) == wd_b`` EXACTLY (0.0 and 1.0 in IEEE arithmetic). For u << u0 the
    exp overflows to inf -> s = 0, the right limit, so the warning is silenced
    (as LESRL ``dwm_cli.make_richards``). Vectorised over ``u``.
    """
    wd_a, wd_b, tau, nu, T_tr = (float(v) for v in (wd_a, wd_b, tau, nu, T_tr))
    if not (tau > 0 and nu > 0 and T_tr > 0):
        raise ValueError(f"need tau, nu, T_tr > 0, got {tau}, {nu}, {T_tr}")
    u0 = T_tr * RICHARDS_U0 / RICHARDS_T_TR if u0 is None else float(u0)

    def _s(u):
        with np.errstate(over="ignore"):
            return (1.0 + np.exp(-(u - u0) / tau)) ** (-1.0 / nu)

    s0, s1 = float(_s(0.0)), float(_s(T_tr))
    if not s1 > s0:
        raise ValueError(f"degenerate Richards segment (s(0)={s0}, s(T_tr)={s1})")

    def _segment(u):
        u = np.asarray(u, dtype=float)
        frac = (_s(u) - s0) / (s1 - s0)
        return wd_a + (wd_b - wd_a) * frac

    _segment.T_tr, _segment.u0, _segment.tau, _segment.nu = T_tr, u0, tau, nu
    return _segment


def richards_transition_length(tau: float) -> float:
    """Transition window of a Richards segment with time constant ``tau``: Emily's
    2600 s scaled by ``tau / 322`` (floored at ``RICHARDS_T_TR_MIN``)."""
    return max(RICHARDS_T_TR * float(tau) / RICHARDS_TAU, RICHARDS_T_TR_MIN)


class _RichardsCycleSchedule:
    """ABSOLUTE cyclic training schedule 270 -> 235 -> 270 with Richards transitions.

    One period = hold ``wd_a`` (``t_hold``) -> Richards segment a -> b (``T_tr``)
    -> hold ``wd_b`` (``t_hold``) -> segment b -> a. Callable as ``f(t) -> wd``
    (``takes_base_wd = False``). At ``t == 0.0`` the start phase ~ U[0,
    phase_max) is drawn inside the first ``wd_a`` hold (``phase_max <= t_hold``,
    the ``_CycleTrainSchedule`` rule: f(0) == wd_a == the pinned burn-in wd), and
    with ``dr`` the episode's ``tau`` / ``nu`` / ``t_hold`` are drawn from their
    bands first (``T_tr`` follows ``tau``; then ``phase_max = t_hold``).
    """

    takes_base_wd = False

    def __init__(self, seed: int, wd_a: float = 270.0, wd_b: float = 235.0,
                 t_hold: float = RICHARDS_T_HOLD, tau: float = RICHARDS_TAU,
                 nu: float = RICHARDS_NU, phase_max: float | None = None, dr=None):
        self._rng = np.random.default_rng(seed)
        self._wd_a, self._wd_b = float(wd_a), float(wd_b)
        self._dr = None if dr is None else {k: (float(v[0]), float(v[1])) for k, v in dr.items()}
        phase_max = float(t_hold) if phase_max is None else float(phase_max)
        if not (0.0 <= phase_max <= float(t_hold)):
            raise ValueError(
                f"phase_max must lie in [0, t_hold={t_hold}] so that f(0) == wd_a "
                f"(no jump at the burn-in boundary); got {phase_max}"
            )
        self._phase_max = phase_max
        self.phase = 0.0
        self._build(float(t_hold), float(tau), float(nu))

    def _build(self, t_hold: float, tau: float, nu: float) -> None:
        self.t_hold, self.tau, self.nu = t_hold, tau, nu
        self.T_tr = richards_transition_length(tau)
        self._seg_ab = make_richards_segment(self._wd_a, self._wd_b, tau, nu, self.T_tr)
        self._seg_ba = make_richards_segment(self._wd_b, self._wd_a, tau, nu, self.T_tr)
        self.period = 2.0 * (t_hold + self.T_tr)

    def _draw(self) -> None:
        rng = self._rng
        phase_max = self._phase_max
        if self._dr is not None:
            t_hold = float(rng.uniform(*self._dr["t_hold"]))
            tau = float(rng.uniform(*self._dr["tau"]))
            nu = float(rng.uniform(*self._dr["nu"]))
            self._build(t_hold, tau, nu)
            phase_max = t_hold
        self.phase = float(rng.uniform(0.0, phase_max)) if phase_max > 0 else 0.0

    def _eval(self, t):
        t = np.asarray(t, dtype=float)
        th, ttr = self.t_hold, self.T_tr
        tau_ = np.mod(t + self.phase, self.period)
        wd = np.full_like(tau_, self._wd_a, dtype=float)
        m = (tau_ >= th) & (tau_ < th + ttr)
        wd[m] = self._seg_ab(tau_[m] - th)
        m = (tau_ >= th + ttr) & (tau_ < 2.0 * th + ttr)
        wd[m] = self._wd_b
        m = tau_ >= 2.0 * th + ttr
        wd[m] = self._seg_ba(tau_[m] - (2.0 * th + ttr))
        return wd

    def __call__(self, t):
        t = float(t)
        if t == 0.0:
            self._draw()
        return float(self._eval(t))


def make_richards_cycle_train(seed: int, dr=None, **kwargs):
    """Build a seeded ABSOLUTE Richards cycle, ``f(t) -> wd`` (see ``_RichardsCycleSchedule``)."""
    return _RichardsCycleSchedule(seed=seed, dr=dr, **kwargs)


class _MixCycleRichards:
    """ABSOLUTE 50/50 mixture: per episode either the R3 linear cycle
    (``cycle_270_235_slow_phase``: 1600 s holds, 320 s ramps) or the DR Richards
    cycle (``richards_270_235_dr_phase``). The coin is flipped at ``t == 0.0`` and
    the chosen branch's own ``t == 0`` redraw fires on the same call; each branch
    has its own sub-seeded RNG, so the untaken branch's stream is not advanced.
    f(0) == 270 either way. ``branch`` names the current episode's branch.
    """

    takes_base_wd = False

    def __init__(self, seed: int, p_cycle: float = 0.5):
        if not 0.0 <= float(p_cycle) <= 1.0:
            raise ValueError(f"p_cycle must be in [0, 1], got {p_cycle}")
        self._rng = np.random.default_rng(seed)
        self._p = float(p_cycle)
        self._cycle = make_cycle_train(270.0, 235.0, 1600.0, 320.0,
                                       seed=int(self._rng.integers(2**31 - 1)), phase_max=1600.0)
        self._richards = make_richards_cycle_train(seed=int(self._rng.integers(2**31 - 1)), dr=RICHARDS_DR)
        self.branch = "cycle"

    @property
    def phase(self) -> float:
        return self._cycle.phase if self.branch == "cycle" else self._richards.phase

    def __call__(self, t):
        t = float(t)
        if t == 0.0:
            self.branch = "cycle" if self._rng.random() < self._p else "richards"
        return float(self._cycle(t) if self.branch == "cycle" else self._richards(t))


class _RichardsWalkSchedule(_DrRampSchedule):
    """``_DrRampSchedule`` (hold / excursion / sign / reflection logic) with every
    linear ramp replaced by a Richards segment: per ramp ``tau ~ U[tau band]``,
    ``nu ~ U[nu band]``, duration ``T_tr = 2600 * tau / 322`` (Emily's window
    scaled), excursion from the parent's magnitude band. RELATIVE (``delta(0) =
    0``), reflected inside ``[wd_lo, wd_hi]`` as the parent.
    """

    def __init__(self, seed: int, tau_lo: float, tau_hi: float, nu_lo: float, nu_hi: float,
                 exc_lo: float, exc_hi: float, wd_lo: float, wd_hi: float,
                 hold_lo: float, hold_hi: float, t_horizon: float = 20000.0):
        if not (tau_lo > 0 and tau_hi >= tau_lo and nu_lo > 0 and nu_hi >= nu_lo):
            raise ValueError(f"bad tau band [{tau_lo}, {tau_hi}] / nu band [{nu_lo}, {nu_hi}]")
        # The parent's rate band only sizes the per-episode draw (n_pairs): the
        # fastest possible mean rate is exc_hi over the shortest transition.
        rate_hi = float(exc_hi) / richards_transition_length(tau_lo)
        super().__init__(seed=seed, rate_lo=rate_hi, rate_hi=rate_hi, exc_lo=exc_lo, exc_hi=exc_hi,
                         wd_lo=wd_lo, wd_hi=wd_hi, hold_lo=hold_lo, hold_hi=hold_hi, t_horizon=t_horizon)
        self._tau_lo, self._tau_hi = float(tau_lo), float(tau_hi)
        self._nu_lo, self._nu_hi = float(nu_lo), float(nu_hi)
        self._segments = []

    def _draw_ramp_shapes(self, n: int):
        taus = self._rng.uniform(self._tau_lo, self._tau_hi, size=n)
        nus = self._rng.uniform(self._nu_lo, self._nu_hi, size=n)
        return np.stack([taus, nus], axis=1)

    def _ramp_duration(self, shape, mag: float) -> float:
        return richards_transition_length(float(shape[0]))

    def _draw(self, base_wd: float) -> None:
        super()._draw(base_wd)
        kt, kd = self._knot_t, self._knot_d
        self._segments = []   # (t_start, t_end, segment g(u) in delta space)
        for i, k in self._ramps:
            tau, nu = (float(v) for v in self._shapes[i])
            t0, t1 = float(kt[k]), float(kt[k + 1])
            self._segments.append((t0, t1, make_richards_segment(kd[k], kd[k + 1], tau, nu, t1 - t0)))
        self._seg_t0 = np.asarray([s[0] for s in self._segments], dtype=float)

    def __call__(self, t, base_wd):
        t = float(t)
        if t == 0.0:
            self._draw(float(base_wd))
        k = int(np.searchsorted(self._seg_t0, t, side="right")) - 1
        if k >= 0:
            t0, t1, seg = self._segments[k]
            if t < t1:
                return float(base_wd) + float(seg(t - t0))
        return float(base_wd) + float(np.interp(t, self._knot_t, self._knot_d))


class _MixStaticRichards:
    """Stage 15 ``mix_static_richards``: RELATIVE ``f(t, base_wd)``; per episode,
    with probability ``p_static`` hold ``base_wd`` (the static control inside the
    preset band), otherwise a ``_RichardsWalkSchedule`` of slow Richards
    excursions reflected inside ``[wd_lo, wd_hi]``. Coin at ``t == 0.0``, the
    walk's own redraw fires on the same call (a static episode never advances the
    walk's RNG). ``branch`` names the current episode's branch.
    """

    def __init__(self, seed: int, p_static: float = 0.5, **walk_kwargs):
        if not 0.0 <= float(p_static) <= 1.0:
            raise ValueError(f"p_static must be in [0, 1], got {p_static}")
        self._rng = np.random.default_rng(seed)
        self._p = float(p_static)
        self._walk = _RichardsWalkSchedule(seed=int(self._rng.integers(2**31 - 1)), **walk_kwargs)
        self.branch = "static"

    def __call__(self, t, base_wd):
        t = float(t)
        if t == 0.0:
            self.branch = "static" if self._rng.random() < self._p else "walk"
        if self.branch == "static":
            return float(base_wd)
        return self._walk(t, base_wd)


def make_mix_static_richards(
    seed: int,
    p_static: float = 0.5,
    tau_lo: float = RICHARDS_DR["tau"][0],
    tau_hi: float = RICHARDS_DR["tau"][1],
    nu_lo: float = RICHARDS_DR["nu"][0],
    nu_hi: float = RICHARDS_DR["nu"][1],
    exc_lo: float = 15.0,
    exc_hi: float = 35.0,
    wd_lo: float = 235.0,
    wd_hi: float = 270.0,
    hold_lo: float = 600.0,
    hold_hi: float = 1600.0,
    **kwargs,
):
    """Build the Stage-15 static / Richards-walk mixture (see ``_MixStaticRichards``)."""
    return _MixStaticRichards(seed, p_static, tau_lo=tau_lo, tau_hi=tau_hi, nu_lo=nu_lo, nu_hi=nu_hi,
                              exc_lo=exc_lo, exc_hi=exc_hi, wd_lo=wd_lo, wd_hi=wd_hi,
                              hold_lo=hold_lo, hold_hi=hold_hi, **kwargs)


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
    # mix_static_ramp: Stage 14 (exp_14 `mix`) -- 70 % of the episodes hold
    # the preset's per-episode base_wd (static control, band 225-315 in
    # les_recipe_band_v2), 30 % run a dr_ramp_narrow-style walk at the Stage-12
    # ramp rate band (0.09-0.14 deg/s, below the 0.1407 deg/s les_3x3 frame
    # slew limit at max_turb_move 30) with 30-45 deg excursions (~ the 35 deg
    # 270 <-> 235 transition). RELATIVE: composes with the preset band.
    "mix_static_ramp": partial(make_mix_static_ramp, p_static=0.7),
    "dr_ramp_slow": partial(make_dr_ramp, rate_lo=0.005, rate_hi=0.015,
                            exc_lo=40.0, exc_hi=50.0),
    # dr_ramp_les: LES-3x3 campaign DR. Brackets the target scenario's
    # 0.0583 deg/s and 35 deg excursion (hold_ramp_270_235) without ever
    # containing the instance. rate_hi = 0.08 is deliberately BELOW the
    # les_3x3 frame slew limit at the campaign's max_turb_move=12
    # (12*360/(2*pi*1533.8)/dt_sim=5 -> 0.0897 deg/s), so the ENTIRE
    # training distribution is frame-tracked: wd_slow == schedule,
    # wd_small == 0, and env.wd is exact and causal everywhere the
    # policy trains.
    "dr_ramp_les": partial(make_dr_ramp, rate_lo=0.03, rate_hi=0.08,
                           exc_lo=25.0, exc_hi=45.0),
    # cycle_270_235_phase: LES-3x3 Stage 5 -- train ON the 270 <-> 235
    # transition. ABSOLUTE (see module docstring / ABSOLUTE_TRAIN_NAMES): the
    # fixed cycle of WD_FUNCTIONS["cycle_270_235"] with a per-episode random
    # start phase ~ U[0, 1000) s. Phase rule: phase_max = t_hold, so every
    # episode starts INSIDE the 270 hold (f(0) == 270 == burn-in wd, no jump at
    # t=0; a full-period phase would start ~58 % of episodes with a <= 35 deg
    # jump that the backward wd_slow pass smears non-causally into the
    # burn-in). The ramp timing is still unlearnable: there is no clock in the
    # observation and the first ramp starts anywhere in [0, 1000] s.
    "cycle_270_235_phase": partial(make_cycle_train, 270.0, 235.0, 1000.0, 200.0,
                                   phase_max=1000.0),
    # cycle_270_235_slow_phase: Stage 6 (IEA22) training schedule -- the _slow
    # cycle (1600 s holds, 320 s ramps, period 3840 s) with a per-episode
    # random start phase ~ U[0, 1600) s. Same hold1-only phase rule as
    # cycle_270_235_phase: phase_max = t_hold, so f(0) == 270 == the pinned
    # burn-in wd (les_recipe_pin270) and there is never a jump at t=0.
    "cycle_270_235_slow_phase": partial(make_cycle_train, 270.0, 235.0, 1600.0,
                                        320.0, phase_max=1600.0),
    # cycle_270_235_305_slow_phase: Stage 12 (exp_12 `sym` / `symveer`) -- the
    # TWO-SIDED _slow cycle 270 -> 235 -> 270 -> 305 -> 270 (1600 s holds, 320 s
    # ramps, period 7680 s, band [235, 305], rate 0.1094 deg/s as the Stage-6..11
    # schedule). Stage 11's one-sided schedule never showed wd > 270, so the
    # policy's yaw sign above 270 was undefined and some seeds toggled at the
    # est-wd jitter around 270. Phase rule: start inside EITHER 270 hold ([0,
    # 1600) or [3840, 5440), equal probability), so f(0) == 270 == the pinned
    # burn-in wd and either excursion can come first.
    "cycle_270_235_305_slow_phase": partial(make_cycle_train, 270.0, 305.0, 1600.0,
                                            320.0, phase_max=1600.0,
                                            levels=(270.0, 235.0, 270.0, 305.0)),
    # ---- Stage 15 (exp_15, cell R 2026-10-19): train on Emily's slow Richards turn ----
    # richards_270_235_phase (`rich`): ABSOLUTE 270 -> 235 -> 270 with Emily's
    # EXACT shape (tau 322, nu 0.39; her 900-3500 s window renormalised to exact
    # endpoints), 1225 s holds, period 7650 s, phase ~ U[0, 1225) inside the 270
    # hold (f(0) == 270 == les_recipe_pin270_v2). Peak 0.0336 deg/s << the 0.1407
    # deg/s frame slew cap at max_turb_move 30: the whole turn is frame-tracked.
    "richards_270_235_phase": partial(make_richards_cycle_train, t_hold=RICHARDS_T_HOLD,
                                      tau=RICHARDS_TAU, nu=RICHARDS_NU),
    # richards_270_235_dr_phase (`rich_dr`): the same cycle with per-episode
    # tau ~ U[160, 640] s, nu ~ U[0.25, 1.0], holds ~ U[800, 1600] s (T_tr = 2600
    # tau / 322 -> 1292-5168 s, peak rate 0.0149-0.0727 deg/s), so the policy
    # cannot learn the clock of one shape.
    "richards_270_235_dr_phase": partial(make_richards_cycle_train, dr=RICHARDS_DR),
    # mix_cycle_richards_phase (`rich_mixcyc`): per episode p 0.5 the R3 linear
    # cycle cycle_270_235_slow_phase (0.1094 deg/s), else the DR Richards cycle:
    # fast + slow turns in one training distribution, f(0) == 270 either way.
    "mix_cycle_richards_phase": partial(_MixCycleRichards, p_cycle=0.5),
    # cycle_stair_phase (`stair`): ABSOLUTE linear stair 270 -> 258 -> 246 -> 235
    # -> 246 -> 258 -> 270, 700 s dwells and 340 s steps (11-12 deg: 0.0324-0.0353
    # deg/s, the Richards peak rate), period 6240 s, phase ~ U[0, 700) in the 270
    # hold. Dwells on BOTH sides of the 250.3 deg wake-alignment sign flip
    # (atan(1420 / 3976)) where lut3 reverses yaw and the Stage-14 arms lagged.
    "cycle_stair_phase": partial(make_cycle_train, 270.0, 235.0, 700.0, 340.0, phase_max=700.0,
                                 levels=(270.0, 258.0, 246.0, 235.0, 246.0, 258.0)),
    # mix_static_richards (`rich_stat`): RELATIVE, for les_recipe_band235_v2
    # (base_wd ~ U[235, 270]). p 0.5 static episodes at base_wd, else 15-35 deg
    # Richards excursions (tau 160-640, nu 0.25-1.0, 600-1600 s holds) REFLECTED
    # inside [235, 270] (its own domain, not the 225-315 default): static
    # coverage of the whole eval band plus slow turns starting anywhere in it.
    "mix_static_richards": partial(make_mix_static_richards, p_static=0.5),
}

# Training schedules that are ABSOLUTE (1-arg ``f(t)``): the preset's wd band
# must be pinned to ``f(0)`` (checked by the trainer) and the env's wd domain
# randomization is replaced, not composed with.
ABSOLUTE_TRAIN_NAMES = frozenset({"cycle_270_235_phase",
                                  "cycle_270_235_slow_phase",
                                  "cycle_270_235_305_slow_phase",
                                  # Stage 15
                                  "richards_270_235_phase",
                                  "richards_270_235_dr_phase",
                                  "mix_cycle_richards_phase",
                                  "cycle_stair_phase"})


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
