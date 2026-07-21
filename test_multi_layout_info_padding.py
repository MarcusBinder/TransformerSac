"""
Repro + regression test for the DR-layout AsyncVectorEnv crash:

    ValueError: could not broadcast input array from shape (9,) into shape (8,)

gymnasium's VectorEnv._add_info recurses into dict-valued info keys and
stacks their inner ndarrays across sub-envs. MultiLayoutEnv._pad_info only
padded TOP-LEVEL ndarrays, so the DEL wrapper's nested dicts
(info["loads"] / info["loads_baseline"] = {channel: (n_turb,) array}) went
through unpadded — fine when every env had the same n_turb (square_3x3),
ragged and fatal under domain-randomized layouts (n ~ U[4, 10]).

Run:  uv run python -m pytest TransformerSac/test_multi_layout_info_padding.py -q
"""

from types import SimpleNamespace

import numpy as np
import pytest
import gymnasium as gym

from helpers.multi_layout_env import MultiLayoutEnv

MAX_TURBINES = 10
CHANNELS = ["Bl1Rad0FlpMnt", "TwrBsFA"]


def make_padder(n_turb: int) -> MultiLayoutEnv:
    """MultiLayoutEnv shell with just the attributes _pad_info reads."""
    env = object.__new__(MultiLayoutEnv)
    env.max_turbines = MAX_TURBINES
    env.pad_value = 0.0
    env.current_layout = SimpleNamespace(n_turbines=n_turb)
    return env


def fake_reset_info(n_turb: int) -> dict:
    """Info dict shaped like the real reset under the DEL wrapper."""
    return {
        "yaw angles agent": np.zeros(n_turb),
        "loads": {ch: np.full(n_turb, np.nan) for ch in CHANNELS},
        "loads_baseline": {ch: np.full(n_turb, np.nan) for ch in CHANNELS},
        "loads_ood": np.zeros(n_turb, dtype=bool),
        "del_ratio_by_channel": {ch: float("nan") for ch in CHANNELS},
        "loads_valid": False,
    }


def aggregate(infos: list[dict]) -> dict:
    """Run gymnasium's own cross-env info stacking (what AsyncVectorEnv does)."""
    vec = object.__new__(gym.vector.VectorEnv)
    vec.num_envs = len(infos)
    out = {}
    for i, info in enumerate(infos):
        out = vec._add_info(out, info, i)
    return out


def test_ragged_nested_dicts_survive_vector_aggregation():
    """DR envs with different n_turb must aggregate without a broadcast error."""
    padded = [
        make_padder(n)._pad_info(fake_reset_info(n)) for n in (8, 9, 4)
    ]
    out = aggregate(padded)  # raised ValueError before the fix
    assert set(out) >= {"loads", "loads_baseline", "del_ratio_by_channel"}


def test_nested_per_turbine_arrays_padded_to_max():
    padded = make_padder(8)._pad_info(fake_reset_info(8))
    for key in ("loads", "loads_baseline"):
        for ch in CHANNELS:
            v = padded[key][ch]
            assert isinstance(v, np.ndarray) and v.shape == (MAX_TURBINES,)
    # Scalar-valued dicts and non-arrays pass through untouched
    assert padded["del_ratio_by_channel"] == pytest.approx(
        fake_reset_info(8)["del_ratio_by_channel"], nan_ok=True
    )
    assert padded["loads_valid"] is False


def test_top_level_behavior_unchanged():
    info = fake_reset_info(9)
    padded = make_padder(9)._pad_info(info)
    assert padded["yaw angles agent"].shape == (MAX_TURBINES,)
    assert padded["loads_ood"] == info["loads_ood"].tolist()  # list fallback
