"""
Save/load round-trip tests for TransformerReplayBuffer.

Verifies that a buffer written with save() and read back with load() is
bit-identical (data arrays, counters, profile lookup state), that metadata
survives the round trip, and that incompatible buffers are rejected with
clear errors instead of silently corrupting training data.
"""

import numpy as np
import torch
import pytest

from replay_buffer import TransformerReplayBuffer


MAX_TURBINES = 6
OBS_DIM = 7
ACTION_DIM = 1
N_DIRS = 36
ROTOR_D = 178.3


def _make_buffer(capacity=100, use_profiles=False, n_layouts=2, max_turbines=MAX_TURBINES, obs_dim=OBS_DIM):
    registry = None
    if use_profiles:
        rng = np.random.default_rng(0)
        registry = [
            (rng.standard_normal((max_turbines, N_DIRS)).astype(np.float32),
             rng.standard_normal((max_turbines, N_DIRS)).astype(np.float32))
            for _ in range(n_layouts)
        ]
    return TransformerReplayBuffer(
        capacity=capacity,
        device=torch.device("cpu"),
        rotor_diameter=ROTOR_D,
        max_turbines=max_turbines,
        obs_dim=obs_dim,
        action_dim=ACTION_DIM,
        use_profiles=use_profiles,
        profile_registry=registry,
    )


def _fill(rb, n, use_profiles=False, seed=123):
    rng = np.random.default_rng(seed)
    for i in range(n):
        mask = np.zeros(rb.max_turbines, dtype=bool)
        mask[4:] = True  # last turbines are padding
        kwargs = {}
        if use_profiles:
            kwargs["layout_index"] = int(rng.integers(0, rb._padded_recep.shape[0]))
            kwargs["permutation"] = rng.permutation(rb.max_turbines)
        rb.add(
            obs=rng.standard_normal((rb.max_turbines, rb._obs.shape[2])).astype(np.float32),
            next_obs=rng.standard_normal((rb.max_turbines, rb._obs.shape[2])).astype(np.float32),
            action=rng.standard_normal((rb.max_turbines, ACTION_DIM)).astype(np.float32),
            reward=float(rng.standard_normal()),
            done=bool(i % 10 == 0),
            raw_positions=rng.uniform(0, 2000, (rb.max_turbines, 2)).astype(np.float32),
            attention_mask=mask,
            wind_direction=float(rng.uniform(0, 360)),
            **kwargs,
        )


DATA_ARRAYS = ["_obs", "_next_obs", "_actions", "_rewards", "_dones",
               "_raw_positions", "_attention_mask", "_wind_directions"]


@pytest.mark.parametrize("use_profiles", [False, True])
def test_roundtrip(tmp_path, use_profiles):
    rb = _make_buffer(use_profiles=use_profiles)
    _fill(rb, 50, use_profiles=use_profiles)
    path = str(tmp_path / "buffer.npz")
    meta_in = {"layouts": "HR1,Lillgrund", "seed": 1, "global_step": 50}
    rb.save(path, extra_meta=meta_in)

    rb2 = _make_buffer(use_profiles=use_profiles)
    meta_out = rb2.load(path)

    assert meta_out == meta_in
    assert len(rb2) == 50
    assert rb2.position == 50
    arrays = DATA_ARRAYS + (["_layout_indices", "_permutations"] if use_profiles else [])
    for name in arrays:
        np.testing.assert_array_equal(getattr(rb, name)[:50], getattr(rb2, name)[:50],
                                      err_msg=f"mismatch in {name}")

    # Sampling from the loaded buffer must work and produce correct shapes
    batch = rb2.sample(16)
    assert batch["observations"].shape == (16, MAX_TURBINES, OBS_DIM)
    assert batch["rewards"].shape == (16, 1)
    if use_profiles:
        assert batch["receptivity"].shape == (16, MAX_TURBINES, N_DIRS)


def test_roundtrip_full_wrapped_buffer(tmp_path):
    """A buffer that wrapped around saves all `capacity` transitions."""
    rb = _make_buffer(capacity=40)
    _fill(rb, 55)  # wraps: size=40, position=15
    path = str(tmp_path / "buffer.npz")
    rb.save(path)

    rb2 = _make_buffer(capacity=40)
    rb2.load(path)
    assert len(rb2) == 40
    assert rb2.position == 0  # full buffer: next write restarts at 0
    np.testing.assert_array_equal(rb._obs, rb2._obs)


def test_load_into_larger_capacity(tmp_path):
    """Warmup buffer loads into a bigger training buffer (the main use case)."""
    rb = _make_buffer(capacity=50)
    _fill(rb, 50)
    path = str(tmp_path / "buffer.npz")
    rb.save(path)

    rb2 = _make_buffer(capacity=200)
    rb2.load(path)
    assert len(rb2) == 50
    assert rb2.position == 50  # new transitions append after the loaded ones
    np.testing.assert_array_equal(rb._obs[:50], rb2._obs[:50])


def test_load_rejects_mismatched_config(tmp_path):
    rb = _make_buffer()
    _fill(rb, 10)
    path = str(tmp_path / "buffer.npz")
    rb.save(path)

    with pytest.raises(ValueError, match="obs_dim"):
        _make_buffer(obs_dim=OBS_DIM + 1).load(path)
    with pytest.raises(ValueError, match="max_turbines"):
        _make_buffer(max_turbines=MAX_TURBINES + 2).load(path)
    with pytest.raises(ValueError, match="use_profiles"):
        _make_buffer(use_profiles=True).load(path)
    with pytest.raises(ValueError, match="capacity"):
        _make_buffer(capacity=5).load(path)


def test_load_rejects_profile_mismatch(tmp_path):
    rb = _make_buffer(use_profiles=True, n_layouts=2)
    _fill(rb, 10, use_profiles=True)
    path = str(tmp_path / "buffer.npz")
    rb.save(path)

    with pytest.raises(ValueError, match="n_layouts"):
        _make_buffer(use_profiles=True, n_layouts=3).load(path)
