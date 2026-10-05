"""ActionHistoryWrapper (commanded delta-yaw history -> raw15span per turbine)
and MultiLayoutEnv.action_history (token order, zero-padded)."""
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from helpers.action_history import ActionHistoryWrapper, span_indices  # noqa: E402
from helpers.multi_layout_env import LayoutConfig, MultiLayoutEnv  # noqa: E402


class _Base(gym.Env):
    def __init__(self, n, act_cols=1):
        self.n_turb = n
        self.D = 100.0
        self.observation_space = gym.spaces.Box(-1, 1, shape=(n, 3), dtype=np.float32)
        shape = (n,) if act_cols == 1 else (n, act_cols)
        self.action_space = gym.spaces.Box(-1, 1, shape=shape, dtype=np.float32)
    def reset(self, *, seed=None, options=None):
        return np.zeros((self.n_turb, 3), np.float32), {}
    def step(self, a):
        return np.zeros((self.n_turb, 3), np.float32), 0.0, False, False, {}


def test_span_indices_mirror_raw15span():
    idx = span_indices(60, 15)
    assert np.array_equal(idx, np.round(np.linspace(0, 59, 15)).astype(int))
    assert np.array_equal(span_indices(15, 15), np.arange(15))


def test_zeros_before_first_command_then_newest_last():
    w = ActionHistoryWrapper(_Base(3), n_hist=60, n_feat=15)
    w.reset()
    h = w.action_history
    assert h.shape == (3, 15) and h.dtype == np.float32 and np.all(h == 0)
    w.step(np.array([0.5, -0.5, 1.0]))
    h = w.action_history
    assert np.allclose(h[:, -1], [0.5, -0.5, 1.0])      # newest sample last
    assert np.all(h[:, :-1] == 0)                        # older slots still zero
    for k in range(70):                                  # overfill the ring
        w.step(np.full(3, float(k)))
    h = w.action_history
    assert h[0, -1] == 69.0
    # 15 evenly spaced samples over the last 60 commands (chronological)
    idx = span_indices(60, 15)
    last60 = np.arange(70 - 60, 70, dtype=float)
    assert np.allclose(h[0], last60[idx])


def test_reset_clears_history():
    w = ActionHistoryWrapper(_Base(2))
    w.reset(); w.step(np.ones(2)); w.step(np.ones(2))
    assert np.any(w.action_history != 0)
    w.reset()
    assert np.all(w.action_history == 0)


def test_two_dim_action_uses_yaw_column():
    w = ActionHistoryWrapper(_Base(2, act_cols=2), yaw_col=0)
    w.reset()
    w.step(np.array([[0.3, 9.0], [-0.3, 9.0]]))
    assert np.allclose(w.action_history[:, -1], [0.3, -0.3])
    # flat [yaw_0..yaw_n | derate...] layout: the yaw block comes first
    w.step(np.array([0.7, -0.7, 5.0, 5.0]))
    assert np.allclose(w.action_history[:, -1], [0.7, -0.7])


def test_multilayout_action_history_is_token_order_and_padded():
    n, max_t = 4, 6
    layout = LayoutConfig(name="row", x_pos=np.arange(n, dtype=np.float32) * 300, y_pos=np.zeros(n, np.float32))
    ml = MultiLayoutEnv(layouts=[layout], env_factory=lambda x, y: _Base(len(x)),
                        per_turbine_wrapper=lambda e: ActionHistoryWrapper(e, n_hist=10, n_feat=5),
                        seed=3, shuffle=True, max_turbines=max_t)
    ml.reset(seed=3)
    perm = ml.current_permutation[:n]
    if np.array_equal(perm, np.arange(n)):
        pytest.skip("identity permutation")
    tok_action = np.array([0.1, 0.2, 0.3, 0.4, 0.0, 0.0], np.float32)  # token order, padded
    ml.step(tok_action)
    h = ml.action_history
    assert h.shape == (max_t, 5)
    # the wrapper stores PHYSICAL order (ml.step unshuffles), the property re-permutes
    assert np.allclose(h[:n, -1], tok_action[:n])
    assert np.all(h[n:] == 0)


def test_multilayout_without_wrapper_raises():
    layout = LayoutConfig(name="row", x_pos=np.zeros(2, np.float32), y_pos=np.zeros(2, np.float32))
    ml = MultiLayoutEnv(layouts=[layout], env_factory=lambda x, y: _Base(len(x)),
                        per_turbine_wrapper=lambda e: e, seed=0)
    with pytest.raises(AttributeError, match="ActionHistoryWrapper"):
        ml.action_history
