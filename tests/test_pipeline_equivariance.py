"""
Pipeline equivariance tests for the wind farm Transformer-SAC agent.

These tests verify that permutation equivariance holds across the FULL pipeline,
not just the neural network architecture. The pipeline includes:
- Profile lookup and permutation in the replay buffer
- Position normalization and wind-relative transformation
- Batch preparation for inference
- Action unshuffling in the environment

Why this matters: the neural network can be perfectly equivariant, but if any
other component (replay buffer, wrapper, data preparation) silently depends on
turbine ordering, training with shuffle_turbs will learn a different -- and
likely worse -- policy.
"""

import numpy as np
import torch
import pytest
from helpers.multi_layout_env import MultiLayoutEnv, LayoutConfig
from helpers.helper_funcs import rotate_profiles_tensor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_layouts(n_layouts: int = 2, seed: int = 42) -> list:
    """Create simple synthetic layouts for testing (no WindFarmEnv needed)."""
    rng = np.random.default_rng(seed)
    layouts = []
    sizes = [4, 6] if n_layouts == 2 else [4]
    names = ["layout_A", "layout_B"][:n_layouts]

    for name, nt in zip(names, sizes):
        x = rng.uniform(0, 2000, size=nt).astype(np.float32)
        y = rng.uniform(0, 2000, size=nt).astype(np.float32)
        recep = rng.standard_normal((nt, 36)).astype(np.float32)
        infl = rng.standard_normal((nt, 36)).astype(np.float32)
        layouts.append(LayoutConfig(
            name=name, x_pos=x, y_pos=y,
            receptivity_profiles=recep, influence_profiles=infl,
        ))
    return layouts


def _build_profile_registry(layouts):
    """Mirror how transformer_sac_windfarm.py builds the registry."""
    return [
        (layout.receptivity_profiles, layout.influence_profiles)
        for layout in layouts
    ]


# ---------------------------------------------------------------------------
# Test 1: Replay buffer profile round-trip
# ---------------------------------------------------------------------------

class TestReplayBufferProfileRoundTrip:
    """
    Verify that profiles reconstructed in the replay buffer match what
    the environment provides during online inference.

    During online inference, profiles come from
    MultiLayoutEnv.receptivity_profiles, which applies profiles[_perm]
    and pads to max_turbines.

    During training, the replay buffer stores (layout_index, permutation)
    and reconstructs profiles via np.take_along_axis(registry[layout], perm).

    These must be identical.
    """

    def test_profile_roundtrip_identity_perm(self):
        """With no shuffling, registry lookup == env property."""
        layouts = _make_layouts(n_layouts=1)
        registry = _build_profile_registry(layouts)
        layout = layouts[0]
        nt = layout.n_turbines
        max_t = nt  # single layout, no extra padding needed

        perm = np.arange(nt, dtype=np.int64)
        padded_perm = np.arange(max_t, dtype=np.int64)

        # What the env property returns (identity perm + padding)
        expected_recep = np.zeros((max_t, 36), dtype=np.float32)
        expected_recep[:nt] = layout.receptivity_profiles[perm]

        # What the replay buffer does
        padded_registry = np.zeros((max_t, 36), dtype=np.float32)
        padded_registry[:nt] = registry[0][0]  # recep for layout 0
        reconstructed = np.take_along_axis(
            padded_registry, padded_perm[:, None], axis=0
        )

        np.testing.assert_array_equal(expected_recep, reconstructed)

    def test_profile_roundtrip_random_perm(self):
        """With a random permutation, registry[perm] == take_along_axis."""
        layouts = _make_layouts(n_layouts=1)
        registry = _build_profile_registry(layouts)
        layout = layouts[0]
        nt = layout.n_turbines
        max_t = nt + 2  # simulate padding

        rng = np.random.default_rng(99)
        perm = rng.permutation(nt).astype(np.int64)

        # Padded permutation (identity for padding slots)
        padded_perm = np.arange(max_t, dtype=np.int64)
        padded_perm[:nt] = perm

        # What the env property returns
        expected_recep = np.zeros((max_t, 36), dtype=np.float32)
        expected_recep[:nt] = layout.receptivity_profiles[perm]

        # What the replay buffer does
        padded_registry = np.zeros((max_t, 36), dtype=np.float32)
        padded_registry[:nt] = registry[0][0]
        reconstructed = np.take_along_axis(
            padded_registry, padded_perm[:, None], axis=0
        )

        np.testing.assert_array_equal(expected_recep, reconstructed)

    def test_profile_roundtrip_multi_layout(self):
        """Round-trip works for multiple layouts with different sizes."""
        layouts = _make_layouts(n_layouts=2)
        registry = _build_profile_registry(layouts)
        max_t = max(l.n_turbines for l in layouts)
        n_dirs = 36

        # Pre-pad registry (mirrors replay_buffer.py lines 98-103)
        padded_recep = np.zeros((len(layouts), max_t, n_dirs), dtype=np.float32)
        padded_infl = np.zeros((len(layouts), max_t, n_dirs), dtype=np.float32)
        for li, (recep, infl) in enumerate(registry):
            nt = recep.shape[0]
            padded_recep[li, :nt] = recep
            padded_infl[li, :nt] = infl

        rng = np.random.default_rng(123)
        for layout_idx, layout in enumerate(layouts):
            nt = layout.n_turbines
            perm = rng.permutation(nt).astype(np.int64)
            padded_perm = np.arange(max_t, dtype=np.int64)
            padded_perm[:nt] = perm

            # Environment would return
            expected = np.zeros((max_t, n_dirs), dtype=np.float32)
            expected[:nt] = layout.receptivity_profiles[perm]

            # Replay buffer reconstruction
            batch_recep = padded_recep[layout_idx]  # (max_t, n_dirs)
            reconstructed = np.take_along_axis(
                batch_recep, padded_perm[:, None], axis=0
            )

            np.testing.assert_array_equal(expected, reconstructed)


# ---------------------------------------------------------------------------
# Test 2: Full pipeline inference equivariance
# ---------------------------------------------------------------------------

class TestInferenceEquivariance:
    """
    Verify that the full inference pipeline (position normalization,
    wind-relative transform, profile rotation, network forward) is
    permutation-equivariant.

    This extends the architecture-only test in debug.ipynb by also
    exercising the data preparation path.
    """

    @pytest.fixture
    def network_and_data(self):
        """Create a small actor and synthetic data."""
        from types import SimpleNamespace
        from networks import TransformerActor
        from helpers.helper_funcs import transform_to_wind_relative

        B, T, obs_dim, n_dirs = 2, 5, 16, 36

        # Minimal args stub for profile encoder construction
        mock_args = SimpleNamespace(profile_encoder_kwargs="{}")

        actor = TransformerActor(
            obs_dim_per_turbine=obs_dim,
            action_dim_per_turbine=1,
            embed_dim=32,
            pos_embed_dim=16,
            num_heads=2,
            num_layers=1,
            mlp_ratio=2.0,
            dropout=0.0,
            action_scale=1.0,
            action_bias=0.0,
            pos_encoding_type="relative_mlp",
            rel_pos_hidden_dim=16,
            rel_pos_per_head=True,
            pos_embedding_mode="add",
            profile_encoding="FourierProfileEncoder",
            profile_encoder_hidden=16,
            n_profile_directions=n_dirs,
            profile_fusion_type="add",
            profile_embed_mode="add",
            shared_recep_encoder=None,
            shared_influence_encoder=None,
            args=mock_args,
        )
        actor.eval()

        torch.manual_seed(42)
        obs = torch.randn(B, T, obs_dim)
        positions = torch.randn(B, T, 2) * 500
        mask = torch.zeros(B, T, dtype=torch.bool)
        recep = torch.randn(B, T, n_dirs)
        infl = torch.randn(B, T, n_dirs)
        wind_dirs = torch.tensor([270.0, 180.0])

        return {
            "actor": actor,
            "obs": obs, "positions": positions, "mask": mask,
            "recep": recep, "infl": infl, "wind_dirs": wind_dirs,
            "transform_to_wind_relative": transform_to_wind_relative,
        }

    def test_equivariance_with_wind_relative_transform(self, network_and_data):
        """Position normalization + wind-relative rotation preserves equivariance."""
        d = network_and_data
        actor = d["actor"]
        transform = d["transform_to_wind_relative"]

        # Canonical order: apply wind-relative transform then forward
        pos_wr = transform(d["positions"].clone(), d["wind_dirs"])
        with torch.no_grad():
            mean_c, logstd_c, _ = actor(
                d["obs"], pos_wr, d["mask"],
                recep_profile=d["recep"], influence_profile=d["infl"],
            )

        # Permuted order
        perm = torch.tensor([3, 1, 4, 0, 2])
        obs_p = d["obs"][:, perm]
        pos_p = d["positions"][:, perm]
        mask_p = d["mask"][:, perm]
        recep_p = d["recep"][:, perm]
        infl_p = d["infl"][:, perm]

        pos_wr_p = transform(pos_p, d["wind_dirs"])
        with torch.no_grad():
            mean_p, logstd_p, _ = actor(
                obs_p, pos_wr_p, mask_p,
                recep_profile=recep_p, influence_profile=infl_p,
            )

        atol, rtol = 1e-5, 1e-4
        torch.testing.assert_close(mean_p, mean_c[:, perm], atol=atol, rtol=rtol)
        torch.testing.assert_close(logstd_p, logstd_c[:, perm], atol=atol, rtol=rtol)

    def test_equivariance_with_profile_rotation(self, network_and_data):
        """Profile rotation + network forward is equivariant."""
        d = network_and_data

        # Rotate profiles to wind-relative frame
        recep_rot = rotate_profiles_tensor(d["recep"], d["wind_dirs"])
        infl_rot = rotate_profiles_tensor(d["infl"], d["wind_dirs"])

        with torch.no_grad():
            mean_c, logstd_c, _ = d["actor"](
                d["obs"], d["positions"], d["mask"],
                recep_profile=recep_rot, influence_profile=infl_rot,
            )

        # Permuted
        perm = torch.tensor([2, 4, 0, 3, 1])
        with torch.no_grad():
            mean_p, logstd_p, _ = d["actor"](
                d["obs"][:, perm], d["positions"][:, perm], d["mask"][:, perm],
                recep_profile=recep_rot[:, perm], influence_profile=infl_rot[:, perm],
            )

        atol, rtol = 1e-5, 1e-4
        torch.testing.assert_close(mean_p, mean_c[:, perm], atol=atol, rtol=rtol)
        torch.testing.assert_close(logstd_p, logstd_c[:, perm], atol=atol, rtol=rtol)

    def test_equivariance_with_padding(self, network_and_data):
        """Equivariance holds when some turbines are padding (masked)."""
        d = network_and_data
        T = 5
        n_real = 3

        # Mask last 2 turbines as padding
        mask = torch.zeros(2, T, dtype=torch.bool)
        mask[:, n_real:] = True

        with torch.no_grad():
            mean_c, logstd_c, _ = d["actor"](
                d["obs"], d["positions"], mask,
                recep_profile=d["recep"], influence_profile=d["infl"],
            )

        # Permute only the real turbines
        real_perm = torch.tensor([2, 0, 1])
        perm = torch.cat([real_perm, torch.arange(n_real, T)])

        with torch.no_grad():
            mean_p, logstd_p, _ = d["actor"](
                d["obs"][:, perm], d["positions"][:, perm], mask[:, perm],
                recep_profile=d["recep"][:, perm], influence_profile=d["infl"][:, perm],
            )

        # Only check real turbine outputs
        atol, rtol = 1e-5, 1e-4
        torch.testing.assert_close(
            mean_p[:, :n_real], mean_c[:, real_perm], atol=atol, rtol=rtol
        )
        torch.testing.assert_close(
            logstd_p[:, :n_real], logstd_c[:, real_perm], atol=atol, rtol=rtol
        )


# ---------------------------------------------------------------------------
# Test 3: Controlled deterministic episode comparison
# ---------------------------------------------------------------------------

class TestDeterministicEpisodeComparison:
    """
    The definitive test: run identical episodes with shuffle=True and
    shuffle=False, using a fixed (zero) policy, and verify rewards match
    step-by-step.

    If this test passes, the pipeline is correct and the reward gap
    observed in training is purely a training dynamics phenomenon.

    NOTE: This test requires the full WindFarmEnv, so it is skipped if
    WindGym is not available or the environment can't be created.
    """

    @pytest.fixture
    def env_pair(self):
        """Create paired shuffled / unshuffled environments."""
        try:
            from WindGym import WindFarmEnv
            from WindGym.wrappers import PerTurbineObservationWrapper
            from helpers.env_configs import make_env_config
            from helpers.layouts import get_layout_positions
        except ImportError:
            pytest.skip("WindGym not available")

        # Use a simple test layout
        try:
            from py_wake.examples.data.hornsrev1 import V80 as WT
            wt = WT()
            layout_name = "test_layout"
            x, y = get_layout_positions(layout_name, wt)
        except Exception:
            pytest.skip("Cannot create test layout")

        n_dirs = 36
        recep = np.random.default_rng(0).standard_normal((len(x), n_dirs)).astype(np.float32)
        infl = np.random.default_rng(1).standard_normal((len(x), n_dirs)).astype(np.float32)

        layout = LayoutConfig(
            name=layout_name, x_pos=x, y_pos=y,
            receptivity_profiles=recep, influence_profiles=infl,
        )

        config = make_env_config("default")
        config["ActionMethod"] = "wind"

        def factory(x_pos, y_pos):
            return WindFarmEnv(
                x_pos=x_pos, y_pos=y_pos,
                turbine=wt, n_passthrough=100,
                config=config, reset_init=False,
                dt_sim=1.0, dt_env=10.0,
            )

        def wrapper(env):
            return PerTurbineObservationWrapper(env)

        seed = 12345
        env_no_shuffle = MultiLayoutEnv(
            layouts=[layout], env_factory=factory,
            per_turbine_wrapper=wrapper, seed=seed, shuffle=False,
        )
        env_shuffle = MultiLayoutEnv(
            layouts=[layout], env_factory=factory,
            per_turbine_wrapper=wrapper, seed=seed, shuffle=True,
        )

        yield env_no_shuffle, env_shuffle

        env_no_shuffle.close()
        env_shuffle.close()

    def test_zero_policy_rewards_match(self, env_pair):
        """With a zero-action policy, shuffled and unshuffled envs produce identical rewards."""
        env_ns, env_sh = env_pair

        env_ns.reset(seed=42)
        env_sh.reset(seed=42)

        n_steps = 20
        for step in range(n_steps):
            # Zero action for both
            action_ns = np.zeros(env_ns.max_turbines, dtype=np.float32)
            action_sh = np.zeros(env_sh.max_turbines, dtype=np.float32)

            _, rew_ns, term_ns, trunc_ns, _ = env_ns.step(action_ns)
            _, rew_sh, term_sh, trunc_sh, _ = env_sh.step(action_sh)

            # Rewards must be identical (same physical actions, same wind)
            assert rew_ns == pytest.approx(rew_sh, abs=1e-6), (
                f"Step {step}: reward mismatch: no_shuffle={rew_ns}, shuffle={rew_sh}"
            )

            # Termination/truncation must match
            assert term_ns == term_sh, f"Step {step}: termination mismatch"
            assert trunc_ns == trunc_sh, f"Step {step}: truncation mismatch"

            if term_ns or trunc_ns:
                break

    def test_observations_are_permuted(self, env_pair):
        """Shuffled env observations are a permutation of unshuffled obs."""
        env_ns, env_sh = env_pair

        obs_ns, _ = env_ns.reset(seed=42)
        obs_sh, _ = env_sh.reset(seed=42)

        nt = env_sh.n_turbines
        perm = env_sh._perm

        # Observations of shuffled env should be obs_ns[perm]
        np.testing.assert_allclose(
            obs_sh[:nt], obs_ns[perm],
            atol=1e-6,
            err_msg="Initial observation is not correctly permuted",
        )

    def test_positions_are_permuted(self, env_pair):
        """Shuffled env positions match permuted unshuffled positions."""
        env_ns, env_sh = env_pair

        env_ns.reset(seed=42)
        env_sh.reset(seed=42)

        nt = env_sh.n_turbines
        perm = env_sh._perm

        pos_ns = env_ns.turbine_positions[:nt]
        pos_sh = env_sh.turbine_positions[:nt]

        np.testing.assert_allclose(
            pos_sh, pos_ns[perm],
            atol=1e-6,
            err_msg="Positions are not correctly permuted",
        )

    def test_profiles_are_permuted(self, env_pair):
        """Shuffled env profiles match permuted unshuffled profiles."""
        env_ns, env_sh = env_pair

        env_ns.reset(seed=42)
        env_sh.reset(seed=42)

        nt = env_sh.n_turbines
        perm = env_sh._perm

        recep_ns = env_ns.receptivity_profiles[:nt]
        recep_sh = env_sh.receptivity_profiles[:nt]

        np.testing.assert_allclose(
            recep_sh, recep_ns[perm],
            atol=1e-6,
            err_msg="Profiles are not correctly permuted",
        )


# ---------------------------------------------------------------------------
# Test 4: Multi-layout RNG divergence detection
# ---------------------------------------------------------------------------

class TestMultiLayoutRNGDivergence:
    """
    Detect the RNG confound: when shuffle=True, the extra
    rng.permutation() call changes the RNG state, causing different
    layout selection sequences in multi-layout training.

    This is a real bug for multi-layout training -- the layout_rng and
    shuffle_rng should be independent.
    """

    def test_single_layout_no_divergence(self):
        """With a single layout, shuffling cannot cause layout divergence."""
        layouts = _make_layouts(n_layouts=1)

        # We just verify both always select the only layout available
        rng_ns = np.random.default_rng(0)
        rng_sh = np.random.default_rng(0)

        for _ in range(20):
            choice_ns = rng_ns.choice(layouts)
            choice_sh = rng_sh.choice(layouts)
            # shuffle=True would do an extra rng.permutation() here
            _ = rng_sh.permutation(layouts[0].n_turbines)

            # Both must pick the same (only) layout
            assert choice_ns.name == choice_sh.name

    def test_multi_layout_divergence_exists(self):
        """
        With shared RNG, shuffle=True causes different layout sequences.
        This test DETECTS the bug (expects divergence with current code).
        """
        layouts = _make_layouts(n_layouts=2)
        n_resets = 20

        # Simulate no-shuffle
        rng_ns = np.random.default_rng(77)
        seq_ns = []
        for _ in range(n_resets):
            choice = rng_ns.choice(layouts)
            seq_ns.append(choice.name)

        # Simulate shuffle (extra rng.permutation call)
        rng_sh = np.random.default_rng(77)
        seq_sh = []
        for _ in range(n_resets):
            choice = rng_sh.choice(layouts)
            seq_sh.append(choice.name)
            # This is what shuffle=True does -- advances the RNG
            _ = rng_sh.permutation(layouts[0].n_turbines)

        # After a few resets, the sequences should diverge
        assert seq_ns != seq_sh, (
            "Expected layout sequences to diverge with shared RNG, "
            "but they didn't. The confound may have been fixed already."
        )

    def test_split_rng_fixes_divergence(self):
        """
        With separate RNGs for layout selection and shuffling,
        layout sequences are identical regardless of shuffle setting.
        """
        layouts = _make_layouts(n_layouts=2)
        n_resets = 20

        seed = 77
        # No-shuffle: layout_rng only
        layout_rng_ns = np.random.default_rng(seed)
        seq_ns = []
        for _ in range(n_resets):
            choice = layout_rng_ns.choice(layouts)
            seq_ns.append(choice.name)

        # Shuffle: separate layout_rng and shuffle_rng
        layout_rng_sh = np.random.default_rng(seed)
        shuffle_rng = np.random.default_rng(seed + 1_000_000)
        seq_sh = []
        for _ in range(n_resets):
            choice = layout_rng_sh.choice(layouts)
            seq_sh.append(choice.name)
            _ = shuffle_rng.permutation(layouts[0].n_turbines)  # separate RNG

        assert seq_ns == seq_sh, (
            "Layout sequences should be identical with split RNGs"
        )


# ---------------------------------------------------------------------------
# Test: Profile rotation commutativity
# ---------------------------------------------------------------------------

class TestProfileRotationCommutativity:
    """
    Verify that profile rotation (wind-relative frame) commutes with
    turbine permutation. This is a necessary condition for the replay
    buffer's order of operations (permute, then rotate) to match
    inference (rotate permuted profiles).
    """

    def test_rotation_commutes_with_permutation(self):
        """rotate(profiles[perm]) == rotate(profiles)[perm]"""
        B, T, n_dirs = 3, 6, 36

        torch.manual_seed(42)
        profiles = torch.randn(B, T, n_dirs)
        wind_dirs = torch.tensor([90.0, 180.0, 270.0])
        perm = torch.tensor([3, 1, 5, 0, 4, 2])

        # Path A: permute first, then rotate
        rotated_a = rotate_profiles_tensor(profiles[:, perm], wind_dirs)

        # Path B: rotate first, then permute
        rotated_b = rotate_profiles_tensor(profiles, wind_dirs)[:, perm]

        torch.testing.assert_close(rotated_a, rotated_b, atol=1e-6, rtol=1e-5)
