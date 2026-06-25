"""Procedural wind-farm layout generation (domain randomization, v8).

Single source of truth for ``make_irregular`` (also imported by the offline
``layout_tools.py`` audit/generation utility) plus ``generate_layout_pool``, which
builds a large fixed pool of irregular layouts for domain-randomized training.

Why a fixed pool rather than a fresh layout every episode: with geometric profiles
(n_profile_directions up to 360) the replay buffer looks profiles up from a registry
indexed by layout id, so an unbounded set of runtime layouts would need either
~tens of GB of inline profile storage or a dynamic registry. A large fixed pool
(e.g. 2048) rides the existing registry mechanism unchanged while still giving
~64x the diversity of v4's 8-layout pool — enough to test the "too few layouts to
generalize" hypothesis. See pywake_transfer_v8.sh.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np


def make_irregular(n_turbines, seed, spread_x, spread_y, D, min_dist_D=3.0):
    """Random layout with a minimum-spacing constraint (rejection sampling).

    Identical generator family to the v4 irregular cells. Places ``n_turbines``
    uniformly in a ``spread_x`` x ``spread_y`` (in D) rectangle, rejecting any
    candidate closer than ``min_dist_D`` rotor diameters to an existing turbine.
    """
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for _ in range(20000):
        cx = rng.uniform(0, spread_x) * D
        cy = rng.uniform(0, spread_y) * D
        if all(np.hypot(cx - xi, cy - yi) >= min_dist_D * D for xi, yi in zip(xs, ys)):
            xs.append(cx); ys.append(cy)
        if len(xs) == n_turbines:
            break
    if len(xs) < n_turbines:
        raise RuntimeError(f"could not place {n_turbines} (seed={seed})")
    return np.array(xs), np.array(ys)


def make_cluster(n_turbines, seed, D):
    """One PLayGen ``cluster`` layout with exactly ``n_turbines`` turbines.

    Poisson-disc archetype from NREL's PLayGen (helpers/playgen.py). Unlike
    ``make_irregular`` (a uniform-random cloud), cluster headroom grows with farm size,
    giving the difficulty diversity the v8 irregular pool lacks (see compare_layouts.py).
    PLayGen sets its own native spacing (2.5-7 D), so ``min_dist_D``/spread do not apply.

    PLayGen draws from the GLOBAL numpy RNG, so we seed it for reproducibility and
    restore the caller's RNG state afterwards (never perturb training's global stream).
    Returns ``(x_array, y_array)`` to mirror ``make_irregular``.
    """
    try:
        from . import playgen          # imported as helpers.layout_gen (training)
    except ImportError:
        import playgen                  # imported with helpers/ on sys.path (tests, layout_tools)
    state = np.random.get_state()
    try:
        np.random.seed(seed)
        xy = playgen.PLayGen(layout_style="cluster", N_turbs=int(n_turbines), D=D)()
    finally:
        np.random.set_state(state)
    return xy[:, 0].astype(float), xy[:, 1].astype(float)


# Operating wind rose for these experiments (matches layout_tools.WDIRS = 225..315),
# sampled coarsely for the cheap geometric headroom screen.
HEADROOM_WD = np.arange(225.0, 316.0, 15.0)


def _downwind_unit(wd_deg: float) -> np.ndarray:
    """Unit vector pointing in the direction the wind TRAVELS (downwind).

    Meteorological convention: wd is the direction the wind comes FROM, measured
    clockwise from North (+y). wd=270 (from west) -> wind travels east (+x).
    """
    r = np.radians(wd_deg)
    return np.array([-np.sin(r), -np.cos(r)])


def turbine_wake_involvement(
    x: np.ndarray, y: np.ndarray, D: float,
    wd_list: np.ndarray = HEADROOM_WD,
    max_streamwise_D: float = 10.0,
    wake_halfwidth_D: float = 1.0,
    k_wake: float = 0.04,
) -> np.ndarray:
    """Boolean mask: which turbines take part in >=1 wake interaction over the rose.

    A turbine is "involved" if, for some sampled wind direction, it sits inside the
    (linearly expanding) wake of an upwind turbine OR casts a wake on a downwind one
    within ``max_streamwise_D`` rotor diameters. This is a cheap geometric proxy for
    "has wake-steering headroom" — it has overlap to gain from, without running the
    yaw optimizer. Calibrated against layout_tools.steering_margin in the v8 tests.
    """
    pos = np.column_stack([np.asarray(x, float), np.asarray(y, float)])
    n = len(pos)
    involved = np.zeros(n, dtype=bool)
    if n < 2:
        return involved
    for wd in wd_list:
        d = _downwind_unit(wd)
        delta = pos[None, :, :] - pos[:, None, :]          # delta[i,j] = pos_j - pos_i
        s = delta @ d                                       # streamwise of j rel. to i
        lat = np.linalg.norm(delta - s[..., None] * d[None, None, :], axis=2)
        halfwidth = (wake_halfwidth_D + k_wake * (s / D)) * D
        waked = (s > 0.0) & (s < max_streamwise_D * D) & (lat < halfwidth)
        np.fill_diagonal(waked, False)
        involved |= waked.any(axis=1)   # i casts a wake on some downwind turbine
        involved |= waked.any(axis=0)   # i sits in some upwind turbine's wake
    return involved


def has_wake_headroom(
    x: np.ndarray, y: np.ndarray, D: float,
    min_involved_frac: float = 0.5,
    wd_list: np.ndarray = HEADROOM_WD,
) -> bool:
    """True if at least ``min_involved_frac`` of turbines are in a wake interaction."""
    inv = turbine_wake_involvement(x, y, D, wd_list=wd_list)
    return bool(inv.mean() >= min_involved_frac)


def generate_layout_pool(
    pool_size: int,
    n_lo: int,
    n_hi: int,
    D: float,
    seed: int = 0,
    min_dist_D: float = 3.0,
    screen_headroom: bool = True,
    min_involved_frac: float = 0.5,
    generator: str = "irregular",
) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    """Build a pool of ``pool_size`` procedural layouts for DR training.

    Each layout draws ``n ~ Uniform[n_lo, n_hi]`` turbines. The pool RNG is derived from
    ``seed`` so different training seeds draw different layout sets.

    ``generator`` selects the archetype:
      * ``"irregular"`` (default) — uniform-random cloud via ``make_irregular`` with the
        spread heuristic (``sx = 6 + 1.6 n``, ``sy = 4 + 1.0 n``, in D) that matches
        ``layout_tools.build_pool``. Names ``dr_n{n}_k{k}`` (unchanged for v8 back-compat).
      * ``"cluster"`` — PLayGen Poisson-disc clusters via ``make_cluster`` (own native
        spacing; ``min_dist_D``/spread ignored). Names ``drc_n{n}_k{k}``.

    Returns a list of ``(name, x_pos, y_pos)`` tuples with unique names. Profiles are
    intentionally NOT computed here — the caller attaches them (geometric or pywake)
    exactly as for the fixed named layouts.
    """
    if n_lo > n_hi:
        raise ValueError(f"dr_n_lo ({n_lo}) must be <= dr_n_hi ({n_hi})")
    if generator not in ("irregular", "cluster"):
        raise ValueError(f"unknown generator '{generator}' (expected 'irregular' or 'cluster')")
    name_prefix = "drc" if generator == "cluster" else "dr"
    rng = np.random.default_rng(seed)
    pool: List[Tuple[str, np.ndarray, np.ndarray]] = []
    rejected = 0
    max_attempts = 50 * pool_size  # guard against an impossible (n_lo, spread) combo
    attempts = 0
    while len(pool) < pool_size and attempts < max_attempts:
        attempts += 1
        n = int(rng.integers(n_lo, n_hi + 1))
        # Per-layout seed from the pool RNG keeps generation reproducible per (seed, k).
        layout_seed = int(rng.integers(0, 2**31 - 1))
        if generator == "cluster":
            x, y = make_cluster(n, layout_seed, D)
        else:
            sx, sy = 6.0 + 1.6 * n, 4.0 + 1.0 * n
            x, y = make_irregular(n, layout_seed, sx, sy, D, min_dist_D=min_dist_D)
        # Reject layouts with no wake-steering headroom (no overlap = no potential gain).
        if screen_headroom and not has_wake_headroom(x, y, D, min_involved_frac=min_involved_frac):
            rejected += 1
            continue
        k = len(pool)
        pool.append((f"{name_prefix}_n{n}_k{k}", x, y))
    if len(pool) < pool_size:
        raise RuntimeError(
            f"only generated {len(pool)}/{pool_size} headroom-positive layouts in "
            f"{attempts} attempts (n in [{n_lo},{n_hi}], min_involved_frac={min_involved_frac}); "
            f"loosen the screen or widen the count range."
        )
    if screen_headroom and rejected:
        print(f"[layout_gen] headroom screen rejected {rejected} layouts "
              f"({100*rejected/(rejected+pool_size):.1f}% of candidates)")
    return pool
