"""Unit tests for the v9 VDN critic aggregation (networks.py TransformerCritic).

Verifies that critic_agg="vdn" (per-turbine q_head -> masked-sum) (a) returns a scalar
farm-Q (B,1) like the pooled critic, (b) ignores padded turbines, and (c) gives an
UN-DILUTED per-turbine action gradient vs the standard masked-mean "pool" critic (which
scales each turbine's gradient ~1/N).

Run: python -m pytest TransformerSac/test_critic_vdn.py -q
"""
import torch

from config import Args
from networks import TransformerCritic

B, N, OBS, ACT = 2, 25, 6, 1


def _make_critic(agg):
    torch.manual_seed(0)
    args = Args()
    args.critic_agg = agg
    return TransformerCritic(
        obs_dim_per_turbine=OBS, action_dim_per_turbine=ACT,
        embed_dim=32, num_heads=2, num_layers=1,
        pos_encoding_type="relative_mlp", profile_encoding=None, args=args,
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


def test_vdn_returns_scalar_farm_q():
    crit = _make_critic("vdn")
    obs, action, pos, mask = _inputs(n_pad=5)
    q = crit(obs, action, pos, mask)
    assert q.shape == (B, 1), f"expected (B,1), got {tuple(q.shape)}"


def test_vdn_ignores_padded_turbines():
    crit = _make_critic("vdn")
    obs, action, pos, mask = _inputs(n_pad=5)
    q0 = crit(obs, action, pos, mask)
    # Perturb ONLY padded turbines -> Q must be unchanged.
    obs2, action2 = obs.clone(), action.clone()
    obs2[:, N - 5:] += 10.0
    action2[:, N - 5:] += 10.0
    q1 = crit(obs2, action2, pos, mask)
    assert torch.allclose(q0, q1, atol=1e-5), "padded turbines leaked into the farm-Q"
    # Perturbing a REAL turbine should change Q (sanity that the sum is live).
    obs3 = obs.clone(); obs3[:, 0] += 5.0
    assert not torch.allclose(q0, crit(obs3, action, pos, mask), atol=1e-4)


def test_vdn_gradient_is_undiluted_vs_pool():
    """Same weights, same input: VDN's per-turbine action gradient is far larger than the
    pooled critic's ~1/N-scaled gradient."""
    pool = _make_critic("pool")
    vdn = _make_critic("vdn")
    vdn.load_state_dict(pool.state_dict())  # identical weights -> isolate the agg effect

    obs, action, pos, mask = _inputs(n_pad=5)  # 20 real turbines

    def action_grad_norm(crit):
        a = action.clone().requires_grad_(True)
        q = crit(obs, a, pos, mask).sum()
        (g,) = torch.autograd.grad(q, a)
        real = (~mask).unsqueeze(-1)            # (B,N,1)
        return (g * real).norm().item()

    g_pool = action_grad_norm(pool)
    g_vdn = action_grad_norm(vdn)
    assert g_vdn > 3.0 * g_pool, (
        f"VDN gradient not un-diluted: vdn={g_vdn:.4g} pool={g_pool:.4g} "
        f"(ratio {g_vdn/max(g_pool,1e-12):.2f}, expected >> 1 for N=20 real turbines)"
    )


def test_pool_unchanged_default():
    """Default critic_agg is 'pool' and still returns (B,1)."""
    crit = _make_critic("pool")
    assert crit.critic_agg == "pool"
    obs, action, pos, mask = _inputs(n_pad=3)
    assert crit(obs, action, pos, mask).shape == (B, 1)


if __name__ == "__main__":
    import sys, pytest
    sys.exit(pytest.main([__file__, "-q"]))
