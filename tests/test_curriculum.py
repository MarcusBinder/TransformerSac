"""Stage-17 oracle-LUT reward-shaping curriculum (helpers/curriculum.py).

``r = w * r_env + (1 - w) * sim`` with ``sim = sim_scale * (1 - min(err / 45, 1))``,
``err = mean |yaw - lut(true wd)|`` in PHYSICAL turbine order (``info["yaw angles
agent"]`` = ``fs.windTurbines.yaw``), the reference read at the env's TRUE wd
(never wd_est). ``w`` ramps linearly from w0 to w1 over start..end env steps and
is pushed through ``MultiLayoutEnv.set_curriculum_weight``.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers.curriculum import (  # noqa: E402
    LutCurriculumRewardWrapper,
    curriculum_weight,
    validate_curriculum_args,
)
from helpers.multi_layout_env import LayoutConfig, MultiLayoutEnv  # noqa: E402

WD_GRID = np.array([266.0, 268.0, 270.0, 272.0])
# rows: wd 266 / 268 / 270 / 272 for a 3-turbine farm (row 270 = +10, row 268 = -10)
LUT = np.array([[-20.0, -20.0, 0.0],
                [-10.0, -10.0, 0.0],
                [10.0, 10.0, 0.0],
                [20.0, 20.0, 0.0]])


class _FarmEnv(gym.Env):
    """Fake WindFarmEnv: physical-order yaw in info, true wd attribute, reward 1.7."""

    def __init__(self, n=3, wd=270.0, yaw=None, reward=1.7):
        self.n_turb = n
        self.D = 100.0
        self.wd = wd
        self.wd_est = wd + 3.0            # must never be read by the wrapper
        self.yaw = np.zeros(n) if yaw is None else np.asarray(yaw, float)
        self.r = reward
        self.observation_space = gym.spaces.Box(-1, 1, shape=(n, 3), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1, 1, shape=(n,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        return np.zeros((self.n_turb, 3), np.float32), {}

    def step(self, a):
        return (np.zeros((self.n_turb, 3), np.float32), self.r, False, False,
                {"yaw angles agent": np.array(self.yaw, dtype=float)})


def _wrap(env, weight, **kw):
    return LutCurriculumRewardWrapper(env, wd_grid=WD_GRID, lut=LUT, weight=weight,
                                      sim_scale=0.1, sim_norm_deg=45.0, ref_hold_deg=1.5,
                                      ref_hold_center=270.0, **kw)


def _step(w):
    w.reset()
    return w.step(np.zeros(3, np.float32))


# ---------------------------------------------------------------- reward mixing
def test_weight_one_is_bitwise_the_env_reward():
    r_env = 0.1 + 0.2                       # 0.30000000000000004
    w = _wrap(_FarmEnv(reward=r_env, yaw=[3.0, -7.0, 1.0]), weight=1.0)
    _, r, _, _, info = _step(w)
    assert r == r_env and info["reward_env"] == r_env
    assert info["curriculum_weight"] == 1.0


def test_weight_zero_yaw_on_reference_gives_sim_scale():
    w = _wrap(_FarmEnv(wd=270.0, yaw=[10.0, 10.0, 0.0]), weight=0.0)
    _, r, _, _, info = _step(w)
    assert r == pytest.approx(0.1) and info["curriculum_sim"] == pytest.approx(0.1)
    assert info["curriculum_ref_err_deg"] == pytest.approx(0.0)


def test_weight_zero_error_at_or_beyond_45_deg_gives_zero():
    w = _wrap(_FarmEnv(wd=270.0, yaw=[-35.0, -35.0, -45.0]), weight=0.0)   # err = 45
    _, r, _, _, info = _step(w)
    assert r == 0.0 and info["curriculum_sim"] == 0.0
    w = _wrap(_FarmEnv(wd=270.0, yaw=[-45.0, -45.0, -45.0]), weight=0.0)   # err > 45 clipped
    _, r, _, _, _ = _step(w)
    assert r == 0.0


def test_weight_zero_half_error_gives_half_sim_scale():
    w = _wrap(_FarmEnv(wd=270.0, yaw=[-12.5, -12.5, -22.5]), weight=0.0)   # err = 22.5
    _, r, _, _, info = _step(w)
    assert r == pytest.approx(0.05) and info["curriculum_ref_err_deg"] == pytest.approx(22.5)


def test_intermediate_weight_mixes_linearly():
    w = _wrap(_FarmEnv(reward=2.0, wd=270.0, yaw=[10.0, 10.0, 0.0]), weight=0.25)
    _, r, _, _, info = _step(w)
    assert r == pytest.approx(0.25 * 2.0 + 0.75 * 0.1)
    assert info["reward_env"] == 2.0 and info["curriculum_weight"] == 0.25


def test_set_weight_clips_to_unit_interval_and_marker_attribute_exists():
    w = _wrap(_FarmEnv(), weight=0.5)
    assert w.curriculum_wrapper is True
    w.set_weight(1.7)
    assert w.weight == 1.0
    w.set_weight(-0.2)
    assert w.weight == 0.0


# ---------------------------------------------------------------- reference frame
def test_reference_inside_the_hold_band_is_the_center_row():
    w = _wrap(_FarmEnv(), weight=0.0)
    assert np.array_equal(w.reference_yaw(269.0), LUT[2])          # 268.5 <= wd <= 270 -> row 270
    assert np.array_equal(w.reference_yaw(268.5), LUT[2])
    assert np.array_equal(w.reference_yaw(270.0), LUT[2])


def test_reference_below_the_hold_band_is_interpolated_per_turbine():
    w = _wrap(_FarmEnv(), weight=0.0)
    ref = w.reference_yaw(268.4)
    expect = LUT[1] + (LUT[2] - LUT[1]) * (268.4 - 268.0) / 2.0
    assert np.allclose(ref, expect) and not np.array_equal(ref, LUT[2])


def test_reference_above_the_center_is_interpolated_not_held():
    w = _wrap(_FarmEnv(), weight=0.0)
    assert np.allclose(w.reference_yaw(271.0), 0.5 * (LUT[2] + LUT[3]))


def test_hold_band_zero_interpolates_everywhere():
    w = LutCurriculumRewardWrapper(_FarmEnv(), wd_grid=WD_GRID, lut=LUT, weight=0.0,
                                   sim_scale=0.1, sim_norm_deg=45.0, ref_hold_deg=0.0,
                                   ref_hold_center=270.0)
    assert np.allclose(w.reference_yaw(269.0), 0.5 * (LUT[1] + LUT[2]))


def test_reference_clamps_to_the_grid_ends():
    w = _wrap(_FarmEnv(), weight=0.0)
    assert np.array_equal(w.reference_yaw(200.0), LUT[0])
    assert np.array_equal(w.reference_yaw(300.0), LUT[3])


def test_reference_reads_the_true_wd_not_the_estimate():
    env = _FarmEnv(wd=270.0, yaw=[10.0, 10.0, 0.0])
    env.wd_est = 266.0                      # LUT row 266 would give err 30
    w = _wrap(env, weight=0.0)
    _, r, _, _, info = _step(w)
    assert info["curriculum_ref_err_deg"] == pytest.approx(0.0) and r == pytest.approx(0.1)


# ---------------------------------------------------------------- physical order / MultiLayoutEnv
def _layout(n=3):
    return LayoutConfig(name="row", x_pos=np.arange(n, dtype=np.float32) * 300, y_pos=np.zeros(n, np.float32))


def _multi(seed, shuffle, weight=0.0, yaw=(10.0, 10.0, 0.0)):
    return MultiLayoutEnv(layouts=[_layout()],
                          env_factory=lambda x, y: _FarmEnv(len(x), wd=270.0, yaw=list(yaw)),
                          per_turbine_wrapper=lambda e: _wrap(e, weight=weight),
                          seed=seed, shuffle=shuffle)


def test_similarity_is_identical_under_two_different_turbine_shuffles():
    sims = []
    for seed in (3, 11):
        ml = _multi(seed, shuffle=True, yaw=(10.0, -4.0, 0.0))
        ml.reset(seed=seed)
        _, r, _, _, info = ml.step(np.zeros(3, np.float32))
        sims.append((float(info["curriculum_sim"]), float(r)))
    assert sims[0] == sims[1]
    assert sims[0][0] == pytest.approx(0.1 * (1 - (14.0 / 3) / 45.0))


def test_set_curriculum_weight_reaches_the_wrapper_through_multi_layout_env():
    ml = _multi(0, shuffle=False, weight=0.0)
    ml.reset(seed=0)
    ml.set_curriculum_weight(1.0)
    _, r, _, _, info = ml.step(np.zeros(3, np.float32))
    assert r == 1.7 and info["curriculum_weight"] == 1.0


def test_set_curriculum_weight_is_reachable_via_get_wrapper_attr_from_an_outer_wrapper():
    ml = _multi(0, shuffle=False, weight=0.0)
    outer = gym.wrappers.TransformReward(ml, lambda r: r)      # stands in for the DR wrapper
    outer.reset(seed=0)
    outer.get_wrapper_attr("set_curriculum_weight")(1.0)
    _, r, _, _, info = outer.step(np.zeros(3, np.float32))
    assert r == 1.7


def test_create_env_reapplies_the_stored_weight_after_a_layout_rebuild():
    ml = _multi(0, shuffle=False, weight=0.0)
    ml.reset(seed=0)
    ml.set_curriculum_weight(1.0)
    ml._create_env(ml.layouts[0])                                # rebuilds the wrapper chain
    _, r, _, _, info = ml.step(np.zeros(3, np.float32))
    assert r == 1.7 and info["curriculum_weight"] == 1.0


def test_set_curriculum_weight_without_the_wrapper_raises():
    ml = MultiLayoutEnv(layouts=[_layout()], env_factory=lambda x, y: _FarmEnv(len(x)),
                        per_turbine_wrapper=lambda e: e, seed=0)
    with pytest.raises(AttributeError, match="curriculum"):
        ml.set_curriculum_weight(0.5)


# ---------------------------------------------------------------- weight schedule
def test_curriculum_weight_is_flat_outside_and_linear_inside():
    assert curriculum_weight(0, 0.0, 1.0, 50_000, 150_000) == 0.0
    assert curriculum_weight(50_000, 0.0, 1.0, 50_000, 150_000) == 0.0
    assert curriculum_weight(100_000, 0.0, 1.0, 50_000, 150_000) == pytest.approx(0.5)
    assert curriculum_weight(150_000, 0.0, 1.0, 50_000, 150_000) == 1.0
    assert curriculum_weight(240_000, 0.0, 1.0, 50_000, 150_000) == 1.0
    assert curriculum_weight(100_000, 0.0, 0.8, 50_000, 150_000) == pytest.approx(0.4)
    assert curriculum_weight(10, 0.3, 0.3, 50_000, 150_000) == 0.3


def test_curriculum_weight_handles_start_equals_end_as_a_step():
    assert curriculum_weight(49_999, 0.0, 1.0, 50_000, 50_000) == 0.0
    assert curriculum_weight(50_000, 0.0, 1.0, 50_000, 50_000) == 1.0


def test_curriculum_weight_rejects_end_before_start():
    with pytest.raises(ValueError):
        curriculum_weight(0, 0.0, 1.0, 150_000, 50_000)


# ---------------------------------------------------------------- validator
def _args(**over):
    base = dict(curriculum="lut", curriculum_w0=0.0, curriculum_w1=1.0, curriculum_start=50_000,
                curriculum_end=150_000, curriculum_sim_scale=0.1, curriculum_sim_norm_deg=45.0,
                curriculum_ref_hold_deg=1.5, curriculum_ref_hold_center=270.0, curriculum_lut_ws=9.5,
                curriculum_lut_ti=0.038, curriculum_lut_sign="farm", reward_scale=1.0)
    base.update(over)
    return SimpleNamespace(**base)


def test_validator_accepts_the_launcher_recipe():
    validate_curriculum_args(_args(), [_layout()], ["row"])
    validate_curriculum_args(_args(curriculum_w1=0.8), [_layout()], ["row"])


@pytest.mark.parametrize("over, layouts, evals", [
    (dict(), [_layout(), _layout()], ["row"]),              # two layouts
    (dict(), [_layout()], ["other"]),                       # eval layout differs
    (dict(curriculum_w0=-0.1), [_layout()], ["row"]),
    (dict(curriculum_w1=1.1), [_layout()], ["row"]),
    (dict(curriculum_start=200_000), [_layout()], ["row"]),  # start > end
    (dict(curriculum_sim_scale=-1.0), [_layout()], ["row"]),
    (dict(curriculum="banana"), [_layout()], ["row"]),
    (dict(curriculum_lut_sign="sideways"), [_layout()], ["row"]),
])
def test_validator_rejects_bad_recipes(over, layouts, evals):
    with pytest.raises(ValueError):
        validate_curriculum_args(_args(**over), layouts, evals)


def test_validator_is_a_no_op_for_curriculum_none():
    validate_curriculum_args(_args(curriculum="none", curriculum_w0=-5.0), [_layout(), _layout()], ["x"])


# ---------------------------------------------------------------- real table (slow)
@pytest.mark.slow
def test_build_curriculum_lut_on_les_3x3_iea22h2_three_bins():
    from helpers.curriculum import build_curriculum_lut
    from helpers.layouts import get_layout_positions
    from helpers.pywake_lut import load_turbine
    wt = load_turbine("iea22h2")
    x, y = get_layout_positions("les_3x3", wt)
    table = build_curriculum_lut(_args(), wt, LayoutConfig(name="les_3x3", x_pos=x, y_pos=y),
                                 wd_grid=np.array([268.0, 270.0, 272.0]))
    assert table["lut"].shape == (3, 9) and np.all(np.abs(table["lut"]) <= 45.0)
    assert np.array_equal(table["wd_grid"], [268.0, 270.0, 272.0])
    # at 270 the upstream (west) column steers; farm sign makes its lead turbines >= 0
    lead = table["lut"][1][np.asarray(x) <= np.min(x) + 1e-6]
    assert np.all(lead >= 0.0) and np.any(lead > 1.0)


# ---------------------------------------------------------------- push rule (review finding 1)
def test_push_rule_lands_exactly_on_w1_under_the_30_step_lattice():
    """global_step advances by num_envs = 30, so w moves in 3e-4 steps and a plain |dw| >= 1e-3 rule
    leaves the last push at 0.9994 (w1 = 1) / 0.79904 (w1 = 0.8): the bitwise r_env branch never fires."""
    from helpers.curriculum import curriculum_push_needed
    for w0, w1 in ((0.0, 1.0), (0.0, 0.8), (0.3, 0.3)):
        pushed, n = None, 0
        for step in range(0, 240_000 + 1, 30):
            w = curriculum_weight(step, w0, w1, 50_000, 150_000)
            if curriculum_push_needed(w, pushed, w0, w1):
                pushed, n = w, n + 1
        assert pushed == w1, (w0, w1, pushed)
        assert n <= 1100
    # the first iteration always pushes, and nothing is pushed while w is unchanged
    assert curriculum_push_needed(0.0, None, 0.0, 1.0) is True
    assert curriculum_push_needed(0.5, 0.5, 0.0, 1.0) is False
    assert curriculum_push_needed(0.5004, 0.5, 0.0, 1.0) is False        # below the 1e-3 threshold mid-ramp
    assert curriculum_push_needed(1.0, 0.9994, 0.0, 1.0) is True          # the endpoint always lands
