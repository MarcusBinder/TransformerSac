"""Unit tests for the PPO shared actor-critic trunk (--ppo_share_trunk).

Locks in the two contracts the feature depends on:
  (a) the networks.py forward()/forward_trunk() split is a pure refactor —
      forward() output is bit-identical to forward_trunk() + action heads,
      and the actor state_dict layout is unchanged (obs_encoder.0.weight
      still present: eval/interp tooling probes that exact key);
  (b) SharedTrunkValue registers ONLY the value head — its state_dict /
      parameters() never contain actor weights (else checkpoints silently
      bloat and Adam sees duplicated params).

Run: cd TransformerSac && python -m pytest tests/test_ppo_shared_trunk.py -q
"""
import torch

from config import Args
from networks import TransformerActor, LOG_STD_MIN, LOG_STD_MAX
from transformer_ppo_windfarm import SharedTrunkValue, ValueHead

B, N, OBS, ACT = 2, 25, 6, 1


def _make_actor():
    torch.manual_seed(0)
    return TransformerActor(
        obs_dim_per_turbine=OBS, action_dim_per_turbine=ACT,
        embed_dim=32, num_heads=2, num_layers=1,
        pos_encoding_type="relative_mlp", profile_encoding=None, args=Args(),
    ).eval()


def _inputs(n_pad=0):
    torch.manual_seed(1)
    obs = torch.randn(B, N, OBS)
    positions = torch.randn(B, N, 2)
    mask = torch.zeros(B, N, dtype=torch.bool)  # True = padding
    if n_pad:
        mask[:, N - n_pad:] = True
    return obs, positions, mask


def test_forward_equals_trunk_plus_heads():
    """networks.py refactor is bit-identical: forward == forward_trunk + heads."""
    actor = _make_actor()
    obs, pos, mask = _inputs(n_pad=5)
    mean, log_std, _ = actor(obs, pos, mask)

    h, _ = actor.forward_trunk(obs, pos, mask)
    mean2 = actor.fc_mean(h)
    log_std2 = torch.tanh(actor.fc_logstd(h))
    log_std2 = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std2 + 1)

    assert torch.equal(mean, mean2), "forward() mean deviates from trunk+head"
    assert torch.equal(log_std, log_std2), "forward() log_std deviates from trunk+head"


def test_actor_state_dict_layout_unchanged():
    """Eval/interp tooling probes actor_state_dict['obs_encoder.0.weight']."""
    sd = _make_actor().state_dict()
    assert "obs_encoder.0.weight" in sd
    assert not any(k.startswith("trunk.") for k in sd), \
        "trunk must stay a method split, not a submodule (would rename all keys)"


def test_value_shape_both_aggs():
    actor = _make_actor()
    obs, pos, mask = _inputs(n_pad=5)
    for agg in ("pool", "vdn"):
        vf = SharedTrunkValue(actor, agg=agg)
        v = vf(obs, pos, mask)
        assert v.shape == (B, 1), f"agg={agg}: expected (B,1), got {tuple(v.shape)}"


def test_padded_turbine_invariance():
    actor = _make_actor()
    obs, pos, mask = _inputs(n_pad=5)
    for agg in ("pool", "vdn"):
        vf = SharedTrunkValue(actor, agg=agg)
        v0 = vf(obs, pos, mask)
        obs2 = obs.clone()
        obs2[:, N - 5:] += 10.0  # perturb ONLY padded turbines
        v1 = vf(obs2, pos, mask)
        assert torch.allclose(v0, v1, atol=1e-5), f"agg={agg}: padding leaked into V"
        obs3 = obs.clone()
        obs3[:, 0] += 5.0  # a REAL turbine must move V (sanity)
        assert not torch.allclose(v0, vf(obs3, pos, mask), atol=1e-4)


def test_state_dict_is_head_only():
    actor = _make_actor()
    vf = SharedTrunkValue(actor, agg="pool")
    keys = set(vf.state_dict().keys())
    assert keys == {"head.v_head.0.weight", "head.v_head.0.bias",
                    "head.v_head.2.weight", "head.v_head.2.bias"}, \
        f"actor leaked into SharedTrunkValue state_dict: {sorted(keys)}"
    head_param_ids = {id(p) for p in vf.parameters()}
    actor_param_ids = {id(p) for p in actor.parameters()}
    assert not head_param_ids & actor_param_ids


def _grads(vf, actor, detach):
    actor.zero_grad(set_to_none=True)
    vf.zero_grad(set_to_none=True)
    obs, pos, mask = _inputs(n_pad=5)
    vf(obs, pos, mask).sum().backward()
    trunk_has = any(p.grad is not None for p in actor.parameters())
    head_has = all(p.grad is not None for p in vf.parameters())
    return trunk_has, head_has


def test_value_gradients_respect_detach_trunk():
    actor = _make_actor()
    trunk_has, head_has = _grads(SharedTrunkValue(actor), actor, detach=False)
    assert trunk_has and head_has, "value backward must reach trunk when not detached"
    trunk_has, head_has = _grads(SharedTrunkValue(actor, detach_trunk=True), actor, detach=True)
    assert head_has and not trunk_has, "detach_trunk=True must stop grads at the trunk"


def test_combined_policy_value_backward():
    """pg_loss + vf_coef*v_loss backward through the double trunk forward runs."""
    actor = _make_actor()
    vf = SharedTrunkValue(actor)
    obs, pos, mask = _inputs(n_pad=5)
    mean, log_std, _ = actor(obs, pos, mask)
    pg_loss = (mean ** 2 + log_std ** 2).mean()
    v_loss = (vf(obs, pos, mask) ** 2).mean()
    (pg_loss + 0.5 * v_loss).backward()
    assert all(p.grad is not None for p in vf.parameters())
    assert actor.fc_mean.weight.grad is not None
    assert actor.obs_encoder[0].weight.grad is not None  # trunk got both losses


def test_optimizer_dedup_param_list():
    """The trainer's id()-dedup over actor+vf params = actor params + head params."""
    actor = _make_actor()
    vf = SharedTrunkValue(actor)
    seen, params = set(), []
    for p in list(actor.parameters()) + list(vf.parameters()):
        if id(p) not in seen:
            seen.add(id(p))
            params.append(p)
    n_actor = sum(1 for _ in actor.parameters())
    n_head = sum(1 for _ in vf.parameters())
    assert len(params) == n_actor + n_head


def test_value_head_standalone_agg():
    """ValueHead itself handles mask=None and both aggs on raw embeddings."""
    torch.manual_seed(2)
    h = torch.randn(B, N, 32)
    for agg in ("pool", "vdn"):
        head = ValueHead(32, agg=agg)
        assert head(h).shape == (B, 1)
        assert head(h, torch.zeros(B, N, dtype=torch.bool)).shape == (B, 1)


if __name__ == "__main__":
    import sys, pytest
    sys.exit(pytest.main([__file__, "-q"]))
