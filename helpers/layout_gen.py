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


def make_grid(n_lo, n_hi, seed, D, spacing_lo_D=4.0, spacing_hi_D=7.0, jitter_D=0.0,
              return_spacing=False):
    """One random rotated nx x ny grid layout with n_lo <= nx*ny <= n_hi turbines.

    Regular-grid archetype for the change_wd sweep: real farms are mostly (rotated)
    grids, and the held-out eval layout (square_5x5, 6D spacing) is one. The count
    is set by sampling (nx, ny) directly — uniformly over all factor pairs with
    nx, ny >= 2 whose product lies in [n_lo, n_hi] — NOT by factoring a pre-sampled
    n (many n in the range are prime and have no >=2x>=2 factorization).
    Per-axis spacing sx, sy ~ U[spacing_lo_D, spacing_hi_D] (in D, independent),
    then the centred grid is rotated by theta ~ U[0, 2pi). ``jitter_D`` adds optional
    per-turbine uniform jitter (default 0 = pure grids). With spacing >= 4D the
    3D min-spacing constraint used elsewhere is always satisfied.
    Returns ``(x_array, y_array)`` to mirror ``make_irregular``; with
    ``return_spacing=True`` also the drawn ``(sx_D, sy_D)`` (Stage 14: the
    ``grid_exclude`` rule of ``generate_layout_pool`` needs them).
    """
    rng = np.random.default_rng(seed)
    shapes = [(nx, ny) for nx in range(2, n_hi + 1) for ny in range(2, n_hi + 1)
              if n_lo <= nx * ny <= n_hi]
    if not shapes:
        raise ValueError(f"no (nx>=2, ny>=2) grid with nx*ny in [{n_lo}, {n_hi}]")
    nx, ny = shapes[rng.integers(len(shapes))]
    sx = rng.uniform(spacing_lo_D, spacing_hi_D) * D
    sy = rng.uniform(spacing_lo_D, spacing_hi_D) * D
    gx, gy = np.meshgrid(np.arange(nx) * sx, np.arange(ny) * sy)
    x, y = gx.ravel().astype(float), gy.ravel().astype(float)
    x -= x.mean(); y -= y.mean()
    theta = rng.uniform(0.0, 2.0 * np.pi)
    c, s = np.cos(theta), np.sin(theta)
    x, y = c * x - s * y, s * x + c * y
    if jitter_D > 0.0:
        x = x + rng.uniform(-jitter_D, jitter_D, size=x.shape) * D
        y = y + rng.uniform(-jitter_D, jitter_D, size=y.shape) * D
    if return_spacing:
        return x, y, sx / D, sy / D
    return x, y


def grid_is_excluded(sx_D: float, sy_D: float, exclude) -> bool:
    """Stage 14 ``grid_held``: True if the drawn per-axis spacings (in D) lie
    within ``tol`` of the excluded pair ``(ex, ey)`` in EITHER ordering (a grid
    is rotated by U[0, 2pi), so 7 x 5 and 5 x 7 are the same farm)."""
    if exclude is None:
        return False
    ex, ey, tol = (float(v) for v in exclude)
    return bool((abs(sx_D - ex) <= tol and abs(sy_D - ey) <= tol)
                or (abs(sx_D - ey) <= tol and abs(sy_D - ex) <= tol))


def parse_grid_exclude(spec):
    """``"sx:sy:tol"`` (rotor diameters, trainer flag ``--dr_grid_exclude_D``)
    -> ``(sx, sy, tol)`` or None for an empty / None spec."""
    if spec is None or str(spec).strip() == "":
        return None
    parts = str(spec).split(":")
    if len(parts) != 3:
        raise ValueError(f"--dr_grid_exclude_D expects 'sx:sy:tol' in D, got {spec!r}")
    sx, sy, tol = (float(v) for v in parts)
    if tol < 0 or sx <= 0 or sy <= 0:
        raise ValueError(f"--dr_grid_exclude_D needs sx, sy > 0 and tol >= 0, got {spec!r}")
    return (sx, sy, tol)


def pool_extent_bounds(pool, D=None):
    """(min, max) over the pool of each layout's max distance from its
    BOUNDING-BOX centre (m) -- exactly the ``max_dist`` of windgym
    ``core/episode_timing.max_wd_step``, whose wd_slow frame-slew cap is
    ``max_turb_move * 360 / (2 pi max_dist)`` deg per sim step. ``D`` is
    accepted for call symmetry with ``generate_layout_pool`` and unused."""
    per = []
    for _, x, y in pool:
        x = np.asarray(x, float); y = np.asarray(y, float)
        cx, cy = (x.max() + x.min()) / 2.0, (y.max() + y.min()) / 2.0
        per.append(float(np.hypot(x - cx, y - cy).max()))
    return min(per), max(per)


def pool_span_bounds(pool, D=None):
    """(min, max) over the pool of each layout's max pairwise turbine distance
    (m) -- exactly the ``max_distance`` of windgym ``core/episode_timing.py``,
    whose reset-sync cap is ``time_max = max_distance * n_passthrough / ws``.
    The min bounds the episode length the SMALLEST pool layout allows."""
    per = []
    for _, x, y in pool:
        d = np.sqrt((x[:, None] - x[None, :]) ** 2 + (y[:, None] - y[None, :]) ** 2)
        per.append(float(d.max()))
    return min(per), max(per)


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
    grid_spacing_lo_D: float = 4.0,
    grid_spacing_hi_D: float = 7.0,
    grid_exclude=None,
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
      * ``"grid"`` — rotated regular grids via ``make_grid``; (nx, ny) is sampled
        inside the generator so n_lo/n_hi bound nx*ny (the loop's per-entry n is
        ignored; names use the actual count). Names ``drg_n{n}_k{k}``. Per-axis
        spacing ~ U[grid_spacing_lo_D, grid_spacing_hi_D] (Stage 14 default 4-7 D as
        before; the exp_14 grid arms use 5-9 D); ``grid_exclude=(sx, sy, tol)`` in D
        re-draws any grid within ``tol`` of that spacing pair in either ordering
        (Stage 14 ``grid_held``: hold out the les_3x3 7 x 5 D farm).
      * ``"mixed"`` — alternates ``grid`` / ``cluster`` per pool entry (even k =
        grid, odd k = cluster); names follow the entry's archetype.

    Returns a list of ``(name, x_pos, y_pos)`` tuples with unique names. Profiles are
    intentionally NOT computed here — the caller attaches them (geometric or pywake)
    exactly as for the fixed named layouts.
    """
    if n_lo > n_hi:
        raise ValueError(f"dr_n_lo ({n_lo}) must be <= dr_n_hi ({n_hi})")
    if generator not in ("irregular", "cluster", "grid", "mixed"):
        raise ValueError(f"unknown generator '{generator}' (expected 'irregular', 'cluster', 'grid' or 'mixed')")
    if not (grid_spacing_lo_D > 0 and grid_spacing_hi_D >= grid_spacing_lo_D):
        raise ValueError(f"bad grid spacing band [{grid_spacing_lo_D}, {grid_spacing_hi_D}] D")
    if grid_exclude is not None and len(grid_exclude) != 3:
        raise ValueError(f"grid_exclude must be (sx_D, sy_D, tol_D), got {grid_exclude!r}")
    rng = np.random.default_rng(seed)
    pool: List[Tuple[str, np.ndarray, np.ndarray]] = []
    rejected = 0
    excluded = 0
    max_attempts = 50 * pool_size  # guard against an impossible (n_lo, spread) combo
    attempts = 0
    while len(pool) < pool_size and attempts < max_attempts:
        attempts += 1
        n = int(rng.integers(n_lo, n_hi + 1))
        # Per-layout seed from the pool RNG keeps generation reproducible per (seed, k).
        layout_seed = int(rng.integers(0, 2**31 - 1))
        # "mixed" alternates the archetype by the NEXT pool index (even = grid, odd =
        # cluster), so a rejected draw re-tries the same archetype.
        gen_k = ("grid", "cluster")[len(pool) % 2] if generator == "mixed" else generator
        name_prefix = {"cluster": "drc", "grid": "drg"}.get(gen_k, "dr")
        if gen_k == "cluster":
            x, y = make_cluster(n, layout_seed, D)
        elif gen_k == "grid":
            # make_grid samples (nx, ny) itself; the loop's n is ignored (grid counts
            # are not uniform over [n_lo, n_hi] — they follow the factor-pair density).
            x, y, sx_D, sy_D = make_grid(n_lo, n_hi, layout_seed, D,
                                         spacing_lo_D=grid_spacing_lo_D,
                                         spacing_hi_D=grid_spacing_hi_D,
                                         return_spacing=True)
            if grid_is_excluded(sx_D, sy_D, grid_exclude):
                excluded += 1
                continue
            n = len(x)
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
    if excluded:
        print(f"[layout_gen] grid_exclude {grid_exclude} re-drew {excluded} grids")
    return pool
