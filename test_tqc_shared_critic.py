"""Unit tests for the shared-trunk TQC critic (networks.py TransformerTQCSharedCritic)
and the TransformerCritic.forward_trunk seam it relies on.

Verifies (a) output contract (n_critics, batch, n_quantiles) in both agg modes,
(b) gradients reach the trunk and every head, (c) heads are diverse at init,
(d) padded turbines don't leak into the quantiles, (e) the forward_trunk refactor
left TransformerCritic.forward bit-identical (manual agg reconstruction, incl. the
zero-action TransformerValue-style call), and (f) the shared critic's param count
is far below the independent ensemble's.

Run: python -m pytest TransformerSac/test_tqc_shared_critic.py -q
"""
import torch

from config import Args
from networks import (
    TransformerCritic,
    TransformerTQCCritic,
    TransformerTQCSharedCritic,
)

B, N, OBS, ACT = 2, 25, 6, 1
N_CRITICS, N_QUANTILES = 5, 25


def _critic_kwargs(agg):
    args = Args()
    args.critic_agg = agg
    return dict(
        obs_dim_per_turbine=OBS, action_dim_per_turbine=ACT,
        embed_dim=32, num_heads=2, num_layers=1,
        pos_encoding_type="relative_mlp", profile_encoding=None, args=args,
    )


def _make_shared(agg):
    torch.manual_seed(0)
    return TransformerTQCSharedCritic(
        n_critics=N_CRITICS, n_quantiles=N_QUANTILES, **_critic_kwargs(agg)
    ).eval()


def _inputs(n_pad=0):
    torch.manual_seed(1)
    obs = torch.randn(B, N, OBS)
    action = torch.randn(B, N, ACT)
    positions = torch.randn(B, N, 2)
    mask = torch.zeros(B, N, dtype=torch.bool)  # True = padding
    if n_pad:
        mask[:, N - n_pad:] = True
    return obs, action, positions, mask


def test_output_shape_both_agg_modes():
    for agg in ("pool", "vdn"):
        crit = _make_shared(agg)
        obs, action, pos, mask = _inputs(n_pad=5)
        q = crit(obs, action, pos, mask)
        assert q.shape == (N_CRITICS, B, N_QUANTILES), \
            f"{agg}: expected {(N_CRITICS, B, N_QUANTILES)}, got {tuple(q.shape)}"


def test_grad_flows_to_trunk_and_every_head():
    crit = _make_shared("pool").train()
    obs, action, pos, mask = _inputs(n_pad=5)
    crit(obs, action, pos, mask).sum().backward()
    trunk_grads = [p.grad for p in crit.trunk.parameters() if p.requires_grad]
    assert trunk_grads and all(g is not None for g in trunk_grads), \
        "trunk has parameters without gradients"
    assert any(g.abs().sum() > 0 for g in trunk_grads), "trunk gradient is all-zero"
    for i, head in enumerate(crit.heads):
        head_grads = [p.grad for p in head.parameters()]
        assert all(g is not None for g in head_grads), f"head {i} missing gradients"
        assert any(g.abs().sum() > 0 for g in head_grads), f"head {i} gradient all-zero"


def test_heads_are_diverse_at_init():
    crit = _make_shared("pool")
    obs, action, pos, mask = _inputs()
    q = crit(obs, action, pos, mask)  # (n_critics, B, n_quantiles)
    for i in range(N_CRITICS):
        for j in range(i + 1, N_CRITICS):
            assert not torch.allclose(q[i], q[j], atol=1e-6), \
                f"heads {i} and {j} produce identical outputs"


def test_padding_invariance():
    for agg in ("pool", "vdn"):
        crit = _make_shared(agg)
        obs, action, pos, mask = _inputs(n_pad=5)
        q0 = crit(obs, action, pos, mask)
        obs2, action2 = obs.clone(), action.clone()
        obs2[:, N - 5:] += 10.0
        action2[:, N - 5:] += 10.0
        q1 = crit(obs2, action2, pos, mask)
        assert torch.allclose(q0, q1, atol=1e-5), \
            f"{agg}: padded turbines leaked into the quantiles"
        # Perturbing a REAL turbine must change the output (sum/pool is live).
        obs3 = obs.clone(); obs3[:, 0] += 5.0
        assert not torch.allclose(q0, crit(obs3, action, pos, mask), atol=1e-4)


def test_forward_trunk_refactor_equivalence():
    """TransformerCritic.forward == agg(forward_trunk(...)) — locks the seam."""
    obs, action, pos, mask = _inputs(n_pad=5)
    for agg in ("pool", "vdn"):
        torch.manual_seed(0)
        crit = TransformerCritic(**_critic_kwargs(agg)).eval()
        with torch.no_grad():
            q_fwd = crit(obs, action, pos, mask)
            h = crit.forward_trunk(obs, action, pos, mask)
            if agg == "vdn":
                q_per = crit.q_head(h) * (~mask).unsqueeze(-1).float()
                q_man = q_per.sum(dim=1)
            else:
                mask_f = (~mask.unsqueeze(-1)).float()
                n_real = mask_f.sum(dim=1).clamp(min=1)
                q_man = crit.q_head((h * mask_f).sum(dim=1) / n_real)
        assert torch.equal(q_fwd, q_man), f"{agg}: forward != agg(forward_trunk)"
    # TransformerValue-style call: zero actions, no mask (the PPO path).
    torch.manual_seed(0)
    crit = TransformerCritic(**_critic_kwargs("pool")).eval()
    with torch.no_grad():
        q_fwd = crit(obs, torch.zeros(B, N, ACT), pos, None)
        h = crit.forward_trunk(obs, torch.zeros(B, N, ACT), pos, None)
        q_man = crit.q_head(h.mean(dim=1))
    assert torch.equal(q_fwd, q_man), "zero-action (TransformerValue) path drifted"


def test_param_count_far_below_independent():
    # Production shape (embed_dim=128, 2 layers): the trunk is ~94% of a critic's
    # params. Ratio = 5(T+H)/(T+5H) -> 5 as H/T -> 0; with a 94% trunk it lands
    # at ~4.0x (measured 3.98x). At the tiny embed_dim=32 used elsewhere in this
    # file the heads dominate and the ratio drops to ~3.4x — not representative.
    kwargs = _critic_kwargs("pool")
    kwargs.update(embed_dim=128, num_heads=4, num_layers=2)
    torch.manual_seed(0)
    shared = TransformerTQCSharedCritic(
        n_critics=N_CRITICS, n_quantiles=N_QUANTILES, **kwargs)
    torch.manual_seed(0)
    independent = TransformerTQCCritic(
        n_critics=N_CRITICS, n_quantiles=N_QUANTILES, **kwargs)
    n_shared = sum(p.numel() for p in shared.parameters())
    n_indep = sum(p.numel() for p in independent.parameters())
    assert n_shared < n_indep / 3.5, (
        f"expected shared < independent/3.5 (~4x at production shape): "
        f"shared={n_shared} indep={n_indep} (ratio {n_indep / n_shared:.2f}x)"
    )


if __name__ == "__main__":
    import sys, pytest
    sys.exit(pytest.main([__file__, "-q"]))
