"""helpers.load_reward: one factory for the DEL-surrogate / proxy-zoo reward
wrapper, driven by config.Args (trainers) or a checkpoint args dict (interp)."""

import sys
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest

_TSAC = Path(__file__).resolve().parents[1]
_REPO = _TSAC.parent
for p in (_REPO, _TSAC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from config import Args  # noqa: E402
from helpers.load_reward import (  # noqa: E402
    build_load_reward_wrapper, load_reward_channels, uses_proxy_reward,
)
from del_surrogate.reward_wrapper import DELRewardWrapper  # noqa: E402
from proxy_zoo import ProxyRewardWrapper  # noqa: E402

N = 2


class _StubEnv(gym.Env):
    observation_space = gym.spaces.Box(-np.inf, np.inf, (N, 3), dtype=np.float32)
    action_space = gym.spaces.Box(-1, 1, (2 * N,), dtype=np.float32)
    backend = "dynamiks"
    Baseline_comp = True
    n_turb = N

    def __init__(self):
        self.fs = SimpleNamespace(step_handlers=[])
        self.fs_baseline = SimpleNamespace(step_handlers=[])


def _args(**kw):
    a = Args()
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def test_defaults_build_surrogate_wrapper():
    a = _args(turbtype="IEA34", del_penalty_scale=1.0)
    assert not uses_proxy_reward(a)
    w = build_load_reward_wrapper(_StubEnv(), a)
    assert type(w) is DELRewardWrapper
    assert w.compare == "farm_max" and w.penalty_kind == "hinge"
    assert w.reward_channels == ["Bl1Rad0FlpMnt"]
    assert load_reward_channels(a) == ["Bl1Rad0FlpMnt"]


def test_load_proxies_builds_proxy_wrapper_and_passes_options():
    a = _args(turbtype="IEA34", del_penalty_scale=2.0, del_allowed_increase=0.2,
              load_proxies="p20_ct,p12_thrust_std", load_reward_proxies="p12_thrust_std",
              load_compare="farm_mean", load_penalty="absolute")
    assert uses_proxy_reward(a)
    w = build_load_reward_wrapper(_StubEnv(), a)
    assert isinstance(w, ProxyRewardWrapper)
    assert w.channels == ["p20_ct", "p12_thrust_std"]
    assert w.reward_channels == ["p12_thrust_std"]
    assert (w.compare, w.penalty_kind) == ("farm_mean", "absolute")
    assert w.penalty_scale == 2.0 and w.allowed_increase == 0.2
    assert load_reward_channels(a) == ["p12_thrust_std"]


def test_load_reward_proxies_defaults_to_all_listed():
    a = _args(turbtype="IEA34", load_proxies="p20_ct,p09_thrust_sum")
    assert load_reward_channels(a) == ["p20_ct", "p09_thrust_sum"]
    w = build_load_reward_wrapper(_StubEnv(), a)
    assert w.reward_channels == ["p20_ct", "p09_thrust_sum"]


def test_dict_args_and_limit_kwargs():
    """interp path: checkpoint args dict + randlim / fixed-limit knobs."""
    a = {**vars(Args()), "turbtype": "IEA34", "load_proxies": "p20_ct",
         "load_compare": "per_turbine_max"}
    w = build_load_reward_wrapper(_StubEnv(), a, limit_range=(0.0, 0.3),
                                  limit_obs_ref=0.3, fixed_limit=0.1)
    assert isinstance(w, ProxyRewardWrapper)
    assert w.compare == "per_turbine_max"
    assert w.limit_range == (0.0, 0.3) and w.fixed_limit == 0.1
    assert w.observation_space.shape == (N, 4)


def test_load_compare_applies_to_surrogate_path_too():
    a = _args(turbtype="IEA34", load_compare="farm_mean")
    w = build_load_reward_wrapper(_StubEnv(), a)
    assert type(w) is DELRewardWrapper and w.compare == "farm_mean"


def test_guard_load_proxies_with_explicit_del_channels():
    a = _args(turbtype="IEA34", load_proxies="p20_ct", del_channels="H0FAMnt")
    with pytest.raises(ValueError, match="del_channels"):
        build_load_reward_wrapper(_StubEnv(), a)
