"""EnsembleCritic must be numerically identical to running its critics separately.

The ensemble runs N TransformerCritics as one vmapped batched forward (to cut
GPU kernel launches). These tests lock in that the batched path equals the
per-critic path and that gradients flow back into every critic.
"""
import torch

from config import Args
from networks import EnsembleCritic


def _make_ensemble(n=2, embed_dim=64, dropout=0.0):
    args = Args()  # defaults: profile_encoder_kwargs="{}", critic_agg="pool", attn off
    kw = dict(
        obs_dim_per_turbine=12,
        action_dim_per_turbine=1,
        embed_dim=embed_dim,
        num_heads=4,
        num_layers=2,
        dropout=dropout,
        pos_encoding_type="relative_mlp",
        profile_encoding="FourierProfileEncoder",
        n_profile_directions=180,
        profile_fusion_type="add",
        profile_embed_mode="concat",
        args=args,
    )
    return EnsembleCritic(n, **kw)


def _inputs(B=4, T=25):
    torch.manual_seed(1)
    obs = torch.randn(B, T, 12)
    action = torch.randn(B, T, 1)
    positions = torch.randn(B, T, 2)
    kpm = torch.zeros(B, T, dtype=torch.bool)
    kpm[:, 20:] = True
    recep = torch.randn(B, T, 180)
    infl = torch.randn(B, T, 180)
    return obs, action, positions, kpm, recep, infl


def test_ensemble_matches_per_critic():
    """vmapped batched forward == stacking the per-critic forwards (eval, deterministic)."""
    torch.manual_seed(0)
    ens = _make_ensemble().eval()
    obs, action, positions, kpm, recep, infl = _inputs()

    with torch.no_grad():
        out = ens(obs, action, positions, kpm, recep, infl)  # (2, B, 1)
        ref = torch.stack([c(obs, action, positions, kpm, recep, infl) for c in ens.critics])

    assert out.shape == ref.shape == (2, obs.shape[0], 1)
    assert torch.allclose(out, ref, atol=1e-5), f"max|Δ|={(out - ref).abs().max():.2e}"


def test_ensemble_none_profiles():
    """None profile args (no-profile path) still match and don't break vmap."""
    torch.manual_seed(0)
    ens = _make_ensemble().eval()
    obs, action, positions, kpm, _, _ = _inputs()
    with torch.no_grad():
        out = ens(obs, action, positions, kpm, None, None)
        ref = torch.stack([c(obs, action, positions, kpm, None, None) for c in ens.critics])
    assert torch.allclose(out, ref, atol=1e-5)


def test_per_critic_state_dict_roundtrip():
    """Checkpoint compat: per-critic state_dicts (the on-disk format) load back into
    a fresh ensemble's sub-critics and reproduce the output bit-for-bit."""
    torch.manual_seed(0)
    src = _make_ensemble().eval()
    obs, action, positions, kpm, recep, infl = _inputs()
    with torch.no_grad():
        ref = src(obs, action, positions, kpm, recep, infl)

    # Save/load exactly how the training loop does it (per sub-critic, old keys).
    sd0, sd1 = src.critics[0].state_dict(), src.critics[1].state_dict()
    dst = _make_ensemble().eval()
    dst.critics[0].load_state_dict(sd0)
    dst.critics[1].load_state_dict(sd1)

    with torch.no_grad():
        out = dst(obs, action, positions, kpm, recep, infl)
    assert torch.allclose(out, ref, atol=1e-6)


def test_ensemble_grads_flow_to_all_critics():
    """Backward through the differentiable param-stack reaches every critic."""
    torch.manual_seed(0)
    ens = _make_ensemble()  # train mode
    obs, action, positions, kpm, recep, infl = _inputs()

    ens.zero_grad()
    out = ens(obs, action, positions, kpm, recep, infl)
    out.sum().backward()

    for i, c in enumerate(ens.critics):
        got = any(p.grad is not None and p.grad.abs().sum() > 0 for p in c.parameters())
        assert got, f"critic {i} received no gradients"
