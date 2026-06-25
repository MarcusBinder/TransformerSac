"""Unit tests for the v8 domain-randomization layout generator (helpers/layout_gen.py).

Run: python -m pytest TransformerSac/helpers/test_layout_gen.py -q
"""
import numpy as np
import pytest

from layout_gen import (
    generate_layout_pool,
    make_irregular,
    make_cluster,
    has_wake_headroom,
    turbine_wake_involvement,
)

D = 178.3  # DTU10MW rotor diameter (m)


def _min_pairwise_dist(x, y):
    pts = np.column_stack([x, y])
    d = np.hypot(pts[:, None, 0] - pts[None, :, 0], pts[:, None, 1] - pts[None, :, 1])
    np.fill_diagonal(d, np.inf)
    return d.min()


def test_make_irregular_respects_min_spacing():
    x, y = make_irregular(20, seed=0, spread_x=6 + 1.6 * 20, spread_y=4 + 20, D=D, min_dist_D=3.0)
    assert len(x) == 20
    assert _min_pairwise_dist(x, y) >= 3.0 * D - 1e-6


def test_pool_counts_in_range_and_unique_names():
    pool = generate_layout_pool(64, n_lo=9, n_hi=25, D=D, seed=1)
    assert len(pool) == 64
    names = [p[0] for p in pool]
    assert len(set(names)) == 64
    for name, x, y in pool:
        assert 9 <= len(x) <= 25
        assert len(x) == len(y)
        assert _min_pairwise_dist(x, y) >= 3.0 * D - 1e-6


def test_pool_is_seed_deterministic_and_seed_varying():
    a = generate_layout_pool(16, 9, 25, D, seed=3)
    b = generate_layout_pool(16, 9, 25, D, seed=3)
    c = generate_layout_pool(16, 9, 25, D, seed=4)
    assert all(np.allclose(ax, bx) for (_, ax, _), (_, bx, _) in zip(a, b))
    # Different seed -> different layouts (at least the first differs in count or coords).
    first_a, first_c = a[0], c[0]
    assert (len(first_a[1]) != len(first_c[1])) or (not np.allclose(first_a[1], first_c[1]))


def test_screen_keeps_only_headroom_layouts():
    pool = generate_layout_pool(32, 9, 25, D, seed=2, screen_headroom=True, min_involved_frac=0.5)
    for _, x, y in pool:
        assert has_wake_headroom(x, y, D, min_involved_frac=0.5)


def test_spread_out_layout_has_no_headroom():
    # 4 turbines on a wide 20D grid -> essentially no wake overlap.
    xs = np.array([0.0, 20 * D, 0.0, 20 * D])
    ys = np.array([0.0, 0.0, 20 * D, 20 * D])
    assert turbine_wake_involvement(xs, ys, D).mean() < 0.5
    assert not has_wake_headroom(xs, ys, D, min_involved_frac=0.5)


def test_invalid_range_raises():
    with pytest.raises(ValueError):
        generate_layout_pool(4, n_lo=10, n_hi=5, D=D, seed=0)


def test_invalid_generator_raises():
    with pytest.raises(ValueError):
        generate_layout_pool(4, n_lo=9, n_hi=25, D=D, seed=0, generator="spiral")


# ============================ cluster generator (PLayGen) ============================
def test_make_cluster_exact_count():
    for n in (6, 9, 16, 25):
        x, y = make_cluster(n, seed=0, D=D)
        assert len(x) == n and len(y) == n


def test_make_cluster_seed_deterministic():
    a = make_cluster(16, seed=7, D=D)
    b = make_cluster(16, seed=7, D=D)
    c = make_cluster(16, seed=8, D=D)
    assert np.allclose(a[0], b[0]) and np.allclose(a[1], b[1])
    assert not (np.array_equal(a[0], c[0]) and np.array_equal(a[1], c[1]))


def test_make_cluster_restores_global_rng():
    np.random.seed(123)
    before = np.random.get_state()
    make_cluster(20, seed=999, D=D)
    after = np.random.get_state()
    # PLayGen uses the global RNG; make_cluster must leave it untouched.
    assert before[0] == after[0]
    assert np.array_equal(before[1], after[1])
    assert before[2:] == after[2:]


def test_cluster_pool_counts_names_and_determinism():
    pool = generate_layout_pool(32, n_lo=9, n_hi=25, D=D, seed=1, generator="cluster")
    assert len(pool) == 32
    names = [p[0] for p in pool]
    assert len(set(names)) == 32
    assert all(nm.startswith("drc_n") for nm in names)  # cluster prefix, not v8 "dr_"
    for name, x, y in pool:
        assert 9 <= len(x) <= 25
        assert len(x) == len(y)
    again = generate_layout_pool(32, n_lo=9, n_hi=25, D=D, seed=1, generator="cluster")
    assert all(np.allclose(ax, bx) and np.allclose(ay, by)
               for (_, ax, ay), (_, bx, by) in zip(pool, again))


def test_cluster_pool_passes_headroom_screen():
    pool = generate_layout_pool(24, n_lo=9, n_hi=25, D=D, seed=2,
                                generator="cluster", screen_headroom=True, min_involved_frac=0.5)
    for _, x, y in pool:
        assert has_wake_headroom(x, y, D, min_involved_frac=0.5)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
