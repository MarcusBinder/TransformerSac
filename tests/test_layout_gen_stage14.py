"""Stage-14 (exp_14) layout-pool flags: grid spacing band, 7 x 5 D exclusion,
the alternating ``mixed`` generator and the pool-extent helper used by the
LUMI launcher's reset-sync / frame-slew guards."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers.layout_gen import generate_layout_pool, pool_extent_bounds, pool_span_bounds  # noqa: E402

D = 284.0


def _nn_spacings_D(x, y):
    d = np.sqrt((x[:, None] - x[None, :]) ** 2 + (y[:, None] - y[None, :]) ** 2)
    d[d == 0] = np.inf
    return d


def test_grid_spacing_band_is_honoured():
    pool = generate_layout_pool(64, 4, 12, D, seed=1, generator="grid",
                                grid_spacing_lo_D=5.0, grid_spacing_hi_D=9.0,
                                screen_headroom=False)
    assert len(pool) == 64
    for _, x, y in pool:
        d = _nn_spacings_D(x, y)
        assert d.min() >= 5.0 * D - 1e-6
        # the nearest-neighbour spacing of a grid is its smaller axis spacing
        assert d.min(axis=1).max() <= 9.0 * D + 1e-6


def test_grid_exclude_never_draws_the_les_3x3_spacing():
    pool = generate_layout_pool(400, 9, 9, D, seed=2, generator="grid",
                                grid_spacing_lo_D=5.0, grid_spacing_hi_D=9.0,
                                grid_exclude=(7.0, 5.0, 0.5), screen_headroom=False)
    assert len(pool) == 400
    for _, x, y in pool:
        d = _nn_spacings_D(x, y)
        # axis spacings of a 3x3 grid: the two smallest distinct distances
        s = np.sort(np.unique(np.round(np.sort(d, axis=1)[:, :4].ravel() / D, 3)))[:2]
        sx, sy = float(s.min()), float(s.max())
        assert not (abs(sx - 5.0) <= 0.5 and abs(sy - 7.0) <= 0.5), (sx, sy)


def test_grid_exclude_accepts_both_orderings():
    a = generate_layout_pool(50, 9, 9, D, seed=2, generator="grid",
                             grid_spacing_lo_D=5.0, grid_spacing_hi_D=9.0,
                             grid_exclude=(7.0, 5.0, 0.5), screen_headroom=False)
    b = generate_layout_pool(50, 9, 9, D, seed=2, generator="grid",
                             grid_spacing_lo_D=5.0, grid_spacing_hi_D=9.0,
                             grid_exclude=(5.0, 7.0, 0.5), screen_headroom=False)
    for (_, xa, ya), (_, xb, yb) in zip(a, b):
        np.testing.assert_allclose(xa, xb)
        np.testing.assert_allclose(ya, yb)


def test_grid_without_exclude_does_draw_the_les_3x3_spacing():
    """The exclusion test above is only meaningful if the band actually
    produces 7 x 5 (+-0.5 D) grids when nothing is excluded."""
    pool = generate_layout_pool(400, 9, 9, D, seed=2, generator="grid",
                                grid_spacing_lo_D=5.0, grid_spacing_hi_D=9.0,
                                screen_headroom=False)
    hits = 0
    for _, x, y in pool:
        d = _nn_spacings_D(x, y)
        s = np.sort(np.unique(np.round(np.sort(d, axis=1)[:, :4].ravel() / D, 3)))[:2]
        if abs(s.min() - 5.0) <= 0.5 and abs(s.max() - 7.0) <= 0.5:
            hits += 1
    assert hits > 0


def test_mixed_generator_alternates():
    pool = generate_layout_pool(10, 4, 12, D, seed=3, generator="mixed", screen_headroom=False)
    names = [n.split("_")[0] for n, _, _ in pool]
    assert names == ["drg", "drc"] * 5


def test_unknown_generator_is_rejected():
    with pytest.raises(ValueError):
        generate_layout_pool(2, 4, 12, D, seed=3, generator="hexagon", screen_headroom=False)


def _windgym_max_dist(x, y):
    """windgym core/episode_timing.max_wd_step: max distance from the BOUNDING-BOX
    centre (not the centroid)."""
    cx, cy = (x.max() + x.min()) / 2, (y.max() + y.min()) / 2
    return np.hypot(x - cx, y - cy).max()


def test_pool_extent_bounds_uses_the_windgym_bbox_centre():
    # clusters are asymmetric, so centroid and bbox centre differ: the helper
    # must follow windgym's max_wd_step rule or the slew guard is off
    pool = generate_layout_pool(12, 4, 12, D, seed=4, generator="cluster", screen_headroom=False)
    lo, hi = pool_extent_bounds(pool, D)
    assert 0 < lo <= hi
    per = [_windgym_max_dist(x, y) for _, x, y in pool]
    assert lo == pytest.approx(min(per)) and hi == pytest.approx(max(per))
    centroid = [np.hypot(x - x.mean(), y - y.mean()).max() for _, x, y in pool]
    assert max(centroid) != pytest.approx(hi)       # the two rules really differ here


def test_pool_span_bounds_is_the_windgym_max_pairwise_distance():
    pool = generate_layout_pool(8, 4, 12, D, seed=4, generator="grid", screen_headroom=False)
    lo, hi = pool_span_bounds(pool, D)
    per = [np.sqrt((x[:, None] - x[None, :]) ** 2 + (y[:, None] - y[None, :]) ** 2).max()
           for _, x, y in pool]
    assert lo == pytest.approx(min(per)) and hi == pytest.approx(max(per))
    elo, ehi = pool_extent_bounds(pool, D)
    assert hi <= 2 * ehi + 1e-6
