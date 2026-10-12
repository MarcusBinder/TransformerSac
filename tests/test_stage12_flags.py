"""Stage-12 (exp_12) trainer flags: the seven new ``Args`` fields and the pure
helpers behind them (``helpers/training_utils.py``). Every flag defaults to the
OLD behaviour, so the Stage-11 presets and command lines are byte-identical.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import Args  # noqa: E402


# ---------------------------------------------------------------------------
# T1: config fields default to the old behaviour
# ---------------------------------------------------------------------------

def test_stage12_args_default_to_the_old_behaviour():
    a = Args()
    assert a.bootstrap_truncated is False
    assert a.env_seed_stride == 1
    assert a.alpha_init is None
    assert a.lr_schedule == "none"
    assert a.lr_final_frac == 0.1
    assert a.tqc_drop_per_net is None
    assert a.tqc_atom_targets is False
    # the Stage-11 recipe's neighbours are untouched
    assert a.tqc_top_quantiles_to_drop == 2 and a.autotune is True and a.policy_frequency == 2


# ---------------------------------------------------------------------------
# T2: time-limit truncation keeps its bootstrap under --bootstrap_truncated
# ---------------------------------------------------------------------------

def test_buffer_done_old_behaviour_stores_truncation_as_terminal():
    from helpers.training_utils import buffer_done
    assert buffer_done(False, True, bootstrap_truncated=False) is True
    assert buffer_done(True, False, bootstrap_truncated=False) is True
    assert buffer_done(False, False, bootstrap_truncated=False) is False


def test_buffer_done_bootstraps_through_a_pure_truncation():
    from helpers.training_utils import buffer_done
    assert buffer_done(False, True, bootstrap_truncated=True) is False
    assert buffer_done(True, True, bootstrap_truncated=True) is True
    assert buffer_done(True, False, bootstrap_truncated=True) is True
    assert buffer_done(False, False, bootstrap_truncated=True) is False


def test_buffer_done_accepts_numpy_bools_and_returns_a_python_bool():
    from helpers.training_utils import buffer_done
    out = buffer_done(np.bool_(False), np.bool_(True), bootstrap_truncated=True)
    assert out is False and float(out) == 0.0
    out = buffer_done(np.bool_(False), np.bool_(True), bootstrap_truncated=False)
    assert out is True and float(out) == 1.0


# ---------------------------------------------------------------------------
# T3: per-seed env streams (--env_seed_stride)
# ---------------------------------------------------------------------------

def test_env_seed_base_stride_one_is_the_old_seed():
    from helpers.training_utils import env_seed_base
    for seed in (1, 2, 6, 42):
        assert env_seed_base(seed, 1) == seed


def test_env_seed_base_stride_1000_separates_seeds_1_to_6():
    from helpers.training_utils import env_seed_base
    num_envs = 30
    streams = {s: {env_seed_base(s, 1000) + i for i in range(num_envs)} for s in range(1, 7)}
    for a in streams:
        for b in streams:
            if a != b:
                assert streams[a].isdisjoint(streams[b]), (a, b)
    # and the old stride-1 layout really did share 29 of 30 streams between neighbours
    old = {s: {env_seed_base(s, 1) + i for i in range(num_envs)} for s in (1, 2)}
    assert len(old[1] & old[2]) == num_envs - 1


def test_env_seed_base_rejects_a_stride_below_one():
    from helpers.training_utils import env_seed_base
    with pytest.raises(ValueError, match="env_seed_stride"):
        env_seed_base(1, 0)


# ---------------------------------------------------------------------------
# T4: --alpha_init (autotune only)
# ---------------------------------------------------------------------------

def test_initial_log_alpha_defaults_to_zero_i_e_alpha_one():
    from helpers.training_utils import initial_log_alpha
    assert initial_log_alpha(None, autotune=True) == 0.0


def test_initial_log_alpha_is_the_log_of_alpha_init():
    from helpers.training_utils import initial_log_alpha
    assert initial_log_alpha(0.1, autotune=True) == pytest.approx(np.log(0.1))
    assert np.exp(initial_log_alpha(0.02, autotune=True)) == pytest.approx(0.02)


def test_initial_log_alpha_rejects_non_positive_and_no_autotune():
    from helpers.training_utils import initial_log_alpha
    with pytest.raises(ValueError, match="alpha_init"):
        initial_log_alpha(0.0, autotune=True)
    with pytest.raises(ValueError, match="alpha_init"):
        initial_log_alpha(-1.0, autotune=True)
    with pytest.raises(ValueError, match="autotune"):
        initial_log_alpha(0.1, autotune=False)
    # no alpha_init and no autotune is the old fixed-alpha path: nothing to reject
    assert initial_log_alpha(None, autotune=False) == 0.0


# ---------------------------------------------------------------------------
# T5: TQC truncation per net + atom targets
# ---------------------------------------------------------------------------

def test_tqc_n_keep_default_drops_two_atoms_in_total():
    from helpers.training_utils import tqc_n_keep
    assert tqc_n_keep(5, 25, top_quantiles_to_drop=2, drop_per_net=None) == 123


def test_tqc_n_keep_drop_per_net_is_the_paper_form():
    from helpers.training_utils import tqc_n_keep
    assert tqc_n_keep(5, 25, top_quantiles_to_drop=2, drop_per_net=2) == 115
    assert tqc_n_keep(5, 25, top_quantiles_to_drop=2, drop_per_net=0) == 125
    assert tqc_n_keep(2, 25, top_quantiles_to_drop=2, drop_per_net=5) == 40


def test_tqc_n_keep_rejects_dropping_everything():
    from helpers.training_utils import tqc_n_keep
    with pytest.raises(ValueError, match="tqc_drop_per_net"):
        tqc_n_keep(5, 25, top_quantiles_to_drop=2, drop_per_net=25)
    with pytest.raises(ValueError, match="tqc_drop_per_net"):
        tqc_n_keep(5, 25, top_quantiles_to_drop=2, drop_per_net=-1)
    with pytest.raises(ValueError, match="tqc_top_quantiles_to_drop"):
        tqc_n_keep(5, 25, top_quantiles_to_drop=125, drop_per_net=None)


def _tqc_batch(seed=0, batch=64, n_keep=123):
    g = torch.Generator().manual_seed(seed)
    kept = torch.sort(torch.randn(batch, n_keep, generator=g) * 3.0, dim=1).values
    rewards = torch.rand(batch, 1, generator=g) * 0.1
    dones = (torch.rand(batch, 1, generator=g) < 0.1).float()
    next_log_pi = torch.randn(batch, 1, generator=g)
    alpha = torch.tensor(0.05)
    return kept, rewards, dones, next_log_pi, alpha


def test_tqc_td_target_mean_branch_is_bitwise_the_old_formula():
    from helpers.training_utils import tqc_td_target
    kept, rewards, dones, next_log_pi, alpha = _tqc_batch()
    gamma = 0.99
    truncated_mean = kept.mean(dim=1, keepdim=True)
    old = rewards + (1 - dones) * gamma * (truncated_mean - alpha * next_log_pi)
    new = tqc_td_target(kept, rewards, dones, gamma, alpha, next_log_pi, atom_targets=False)
    assert new.shape == (64, 1)
    assert torch.equal(new, old)


def test_tqc_td_target_atom_branch_keeps_every_atom():
    from helpers.training_utils import tqc_td_target
    kept, rewards, dones, next_log_pi, alpha = _tqc_batch(n_keep=115)
    gamma = 0.99
    tgt = tqc_td_target(kept, rewards, dones, gamma, alpha, next_log_pi, atom_targets=True)
    assert tgt.shape == (64, 115)
    done_rows = dones[:, 0] > 0
    assert done_rows.any() and (~done_rows).any()
    # terminal rows: the target is the reward on every atom
    assert torch.allclose(tgt[done_rows], rewards[done_rows].expand(-1, 115))
    # bootstrapped rows: the per-atom target, same order as the sorted atoms
    expect = rewards + gamma * (kept - alpha * next_log_pi)
    assert torch.allclose(tgt[~done_rows], expect[~done_rows])
    assert torch.all(tgt[~done_rows][:, 1:] >= tgt[~done_rows][:, :-1])


def test_quantile_huber_loss_broadcasts_over_target_atoms():
    from networks import quantile_huber_loss
    g = torch.Generator().manual_seed(1)
    pred = torch.randn(32, 25, generator=g)
    taus = (torch.arange(25).float() + 0.5) / 25
    target = torch.randn(32, 115, generator=g)
    vec = quantile_huber_loss(pred, target, taus)
    manual = torch.stack([quantile_huber_loss(pred, target[:, j:j + 1], taus) for j in range(115)]).mean()
    assert vec.shape == ()
    assert torch.allclose(vec, manual, atol=1e-6)
    # a (batch, 1) target is the old scalar-target call
    assert quantile_huber_loss(pred, target[:, :1], taus).shape == ()


# ---------------------------------------------------------------------------
# T6: --lr_schedule cosine|linear with --lr_final_frac
# ---------------------------------------------------------------------------

def test_lr_multiplier_none_is_identity():
    from helpers.training_utils import lr_multiplier
    for p in (0.0, 0.3, 1.0):
        assert lr_multiplier("none", p, 0.05) == 1.0


def test_lr_multiplier_cosine_endpoints_and_midpoint():
    from helpers.training_utils import lr_multiplier
    assert lr_multiplier("cosine", 0.0, 0.05) == pytest.approx(1.0)
    assert lr_multiplier("cosine", 1.0, 0.05) == pytest.approx(0.05)
    assert lr_multiplier("cosine", 0.5, 0.05) == pytest.approx(0.525)
    # 6e-4 -> 3e-5 is the lrsched arm
    assert 6e-4 * lr_multiplier("cosine", 1.0, 0.05) == pytest.approx(3e-5)


def test_lr_multiplier_linear_endpoints_and_midpoint():
    from helpers.training_utils import lr_multiplier
    assert lr_multiplier("linear", 0.0, 0.1) == pytest.approx(1.0)
    assert lr_multiplier("linear", 0.5, 0.1) == pytest.approx(0.55)
    assert lr_multiplier("linear", 1.0, 0.1) == pytest.approx(0.1)


def test_lr_multiplier_clamps_progress_and_rejects_unknown_schedules():
    from helpers.training_utils import lr_multiplier
    assert lr_multiplier("cosine", -0.5, 0.05) == pytest.approx(1.0)
    assert lr_multiplier("cosine", 1.7, 0.05) == pytest.approx(0.05)
    assert isinstance(lr_multiplier("cosine", 0.25, 0.05), float)
    with pytest.raises(ValueError, match="lr_schedule"):
        lr_multiplier("step", 0.5, 0.05)


def test_lr_progress_spans_the_update_budget():
    from helpers.training_utils import lr_progress
    assert lr_progress(0, 10_000, 200_000) == 0.0
    assert lr_progress(10_000, 10_000, 200_000) == 0.0
    assert lr_progress(105_000, 10_000, 200_000) == pytest.approx(0.5)
    assert lr_progress(200_000, 10_000, 200_000) == 1.0
    assert lr_progress(200_060, 10_000, 200_000) == 1.0   # the two extra iterations clamp


# ---------------------------------------------------------------------------
# CLI contract: the launcher passes bare bools and Optional ints through tyro
# ---------------------------------------------------------------------------

def test_stage12_flags_parse_through_tyro_as_the_launcher_passes_them():
    import tyro
    a = tyro.cli(Args, args=["--bootstrap_truncated", "--env_seed_stride", "1000", "--alpha_init", "0.1",
                             "--policy_frequency", "1", "--tqc_drop_per_net", "2", "--tqc_atom_targets",
                             "--policy_lr", "6e-4", "--q_lr", "6e-4", "--lr_schedule", "cosine", "--lr_final_frac", "0.05",
                             "--utd_ratio", "2", "--learning_starts", "50000", "--total_timesteps", "240000"])
    assert a.bootstrap_truncated is True and a.env_seed_stride == 1000
    assert a.alpha_init == pytest.approx(0.1) and a.policy_frequency == 1
    assert a.tqc_drop_per_net == 2 and a.tqc_atom_targets is True
    assert a.policy_lr == pytest.approx(6e-4) and a.lr_schedule == "cosine" and a.lr_final_frac == pytest.approx(0.05)
    assert a.utd_ratio == 2 and a.learning_starts == 50000 and a.total_timesteps == 240000
    # and the bare base command line (no Stage-12 flags) keeps every default
    b = tyro.cli(Args, args=["--seed", "3"])
    assert (b.bootstrap_truncated, b.env_seed_stride, b.alpha_init, b.lr_schedule, b.tqc_drop_per_net, b.tqc_atom_targets) == \
           (False, 1, None, "none", None, False)


# ---------------------------------------------------------------------------
# Review fix: --alpha_lr decouples the entropy coefficient's lr from --q_lr
# (the lrsched arm doubles q_lr; alpha must stay at the Stage-11 3e-4)
# ---------------------------------------------------------------------------

def test_alpha_lr_defaults_to_q_lr():
    from helpers.training_utils import alpha_learning_rate
    assert Args().alpha_lr is None
    assert alpha_learning_rate(None, 3e-4) == pytest.approx(3e-4)
    assert alpha_learning_rate(None, 6e-4) == pytest.approx(6e-4)


def test_alpha_lr_overrides_q_lr_and_rejects_non_positive():
    from helpers.training_utils import alpha_learning_rate
    assert alpha_learning_rate(3e-4, 6e-4) == pytest.approx(3e-4)
    with pytest.raises(ValueError, match="alpha_lr"):
        alpha_learning_rate(0.0, 6e-4)
    import tyro
    a = tyro.cli(Args, args=["--q_lr", "6e-4", "--alpha_lr", "3e-4"])
    assert a.alpha_lr == pytest.approx(3e-4) and a.q_lr == pytest.approx(6e-4)
