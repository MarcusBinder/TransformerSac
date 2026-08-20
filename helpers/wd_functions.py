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
                 t_ramp: float, phase_max: float):
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
        self._base = make_cycle(wd_a, wd_b, t_hold, t_ramp)
        self.period = self._base.period
        self.phase = 0.0

    def _draw(self) -> None:
        self.phase = float(self._rng.uniform(0.0, self._phase_max)) if self._phase_max > 0 else 0.0

    def __call__(self, t):
        t = float(t)
        if t == 0.0:
            self._draw()
        return float(self._base(t + self.phase))


def make_cycle_train(wd_a: float, wd_b: float, t_hold: float, t_ramp: float,
                     seed: int, phase_max: float):
    """Build a seeded ABSOLUTE cyclic training schedule, ``f(t) -> wd``."""
    return _CycleTrainSchedule(seed=seed, wd_a=wd_a, wd_b=wd_b, t_hold=t_hold,
                               t_ramp=t_ramp, phase_max=phase_max)


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
}

# Training schedules that are ABSOLUTE (1-arg ``f(t)``): the preset's wd band
# must be pinned to ``f(0)`` (checked by the trainer) and the env's wd domain
# randomization is replaced, not composed with.
ABSOLUTE_TRAIN_NAMES = frozenset({"cycle_270_235_phase",
                                  "cycle_270_235_slow_phase"})


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
