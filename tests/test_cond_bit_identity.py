"""Cond-OFF path is bit-identical to main; cond-ON injection modes behave as specified."""
import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import Args  # noqa: E402
from networks import ParamEncoder, TransformerActor  # noqa: E402
from tests._cond_snapshot import OBS_DIM, SNAP, snapshot  # noqa: E402


def _inputs(seed=5, b=2, n=4):
    g = torch.Generator().manual_seed(seed)
    obs = torch.randn(b, n, OBS_DIM, generator=g)
    pos = torch.randn(b, n, 2, generator=g)
    rows = [[False] * (n - 1) + [True], [False] * (n - 2) + [True, True]]
    mask = torch.tensor(rows[:b] if b <= 2 else rows + [[False] * n] * (b - 2))
    e = torch.randn(b, n, 12, generator=g)
    e[mask] = 0.0
    return obs, pos, mask, e


def test_cond_off_matches_main_snapshot():
    ref = json.loads(SNAP.read_text())
    now = snapshot()
    assert now["actor"] == ref["actor"]
    assert now["critic"] == ref["critic"]
    assert now["tqc_shared"] == ref["tqc_shared"]
    assert now["forward"] == ref["forward"]


def test_cond_off_forward_rejects_cond_and_ignores_none():
    actor = TransformerActor(obs_dim_per_turbine=OBS_DIM, args=Args())
    obs, pos, mask, e = _inputs()
    with torch.no_grad():
        a = actor(obs, pos, mask)
        b = actor(obs, pos, mask, cond=None)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    with pytest.raises(ValueError, match="cond"):
        actor(obs, pos, mask, cond=e)
    assert not hasattr(actor, "param_encoder") or actor.param_encoder is None
    assert actor.cond_dim == 0 and actor.cond_on is False


@pytest.mark.parametrize("dz", [0, 8])
def test_concat_cond_on_requires_cond_and_changes_only_obs_encoder_width(dz):
    args = Args(cond_source="turbine_farm", cond_latent_dim=dz, cond_mode="concat")
    actor = TransformerActor(obs_dim_per_turbine=OBS_DIM, args=args)
    want = dz if dz > 0 else 12
    assert actor.cond_on and actor.cond_dim == want
    assert actor.obs_encoder[0].weight.shape == (128, OBS_DIM + want)
    keys = set(actor.state_dict())
    base_keys = set(TransformerActor(obs_dim_per_turbine=OBS_DIM, args=Args()).state_dict())
    extra = keys - base_keys
    if dz > 0:
        assert isinstance(actor.param_encoder, ParamEncoder)
        assert extra and all(k.startswith("param_encoder.") for k in extra)
    else:
        assert actor.param_encoder is None and extra == set()
    obs, pos, mask, e = _inputs()
    with pytest.raises(ValueError, match="cond"):
        actor(obs, pos, mask)
    with torch.no_grad():
        m1, _, _ = actor(obs, pos, mask, cond=e)
        m2, _, _ = actor(obs, pos, mask, cond=e + 0.5)
        z = actor.encode_cond(e)
        m3, _, _ = actor(obs, pos, mask, cond=z, cond_is_latent=True)
    assert m1.shape == (2, 4, 1)
    assert not torch.allclose(m1, m2)            # the policy depends on e
    assert torch.allclose(m1, m3, atol=1e-6)     # latent path == encode then inject
    assert z.shape == (2, 4, want)


def test_concat_zero_padded_weights_reproduce_nominal_actor():
    """Identity-preserving warm start: zero columns for z in obs_encoder.0.weight."""
    torch.manual_seed(0)
    nominal = TransformerActor(obs_dim_per_turbine=OBS_DIM, args=Args())
    cond = TransformerActor(obs_dim_per_turbine=OBS_DIM,
                            args=Args(cond_source="turbine", cond_latent_dim=0, cond_mode="concat"))
    sd = dict(nominal.state_dict())
    w = sd["obs_encoder.0.weight"]
    sd["obs_encoder.0.weight"] = torch.cat([w, torch.zeros(w.shape[0], 12)], dim=1)
    cond.load_state_dict(sd, strict=True)
    obs, pos, mask, e = _inputs()
    with torch.no_grad():
        a = nominal(obs, pos, mask)[0]
        b = cond(obs, pos, mask, cond=e)[0]
    assert torch.allclose(a, b, atol=1e-6)


def test_film_zero_init_is_identity_at_init():
    torch.manual_seed(0)
    nominal = TransformerActor(obs_dim_per_turbine=OBS_DIM, args=Args())
    film = TransformerActor(obs_dim_per_turbine=OBS_DIM,
                            args=Args(cond_source="turbine_farm", cond_latent_dim=8, cond_mode="film"))
    assert film.obs_encoder[0].weight.shape == (128, OBS_DIM)   # concat not used
    for layer in film.transformer.layers:
        assert torch.all(layer.film1.weight == 0) and torch.all(layer.film1.bias == 0)
        assert torch.all(layer.film2.weight == 0) and torch.all(layer.film2.bias == 0)
        assert layer.film1.out_features == 2 * 128
    missing, unexpected = film.load_state_dict(nominal.state_dict(), strict=False)
    assert unexpected == []
    assert all(k.startswith("param_encoder.") or ".film" in k for k in missing)
    obs, pos, mask, e = _inputs()
    with torch.no_grad():
        a = nominal(obs, pos, mask)[0]
        b = film(obs, pos, mask, cond=e)[0]
    assert torch.allclose(a, b, atol=1e-6)
    # ...and FiLM is live once the weights move
    with torch.no_grad():
        for layer in film.transformer.layers:
            layer.film1.weight.fill_(0.1)
        c = film(obs, pos, mask, cond=e)[0]
    assert not torch.allclose(a, c)


def test_cond_padded_tokens_are_equivariant():
    """Permuting tokens (with e) permutes the output; padded rows stay masked."""
    args = Args(cond_source="turbine_farm", cond_latent_dim=8, cond_mode="concat", pos_encoding_type="relative_mlp")
    torch.manual_seed(1)
    actor = TransformerActor(obs_dim_per_turbine=OBS_DIM, args=args)
    obs, pos, mask, e = _inputs(b=1, n=4)
    perm = torch.tensor([2, 0, 1, 3])
    with torch.no_grad():
        m = actor(obs, pos, mask, cond=e)[0]
        mp = actor(obs[:, perm], pos[:, perm], mask[:, perm], cond=e[:, perm])[0]
    assert torch.allclose(m[:, perm][:, :3], mp[:, :3], atol=1e-5)


def test_get_action_threads_cond():
    args = Args(cond_source="turbine_farm", cond_latent_dim=8)
    actor = TransformerActor(obs_dim_per_turbine=OBS_DIM, args=args)
    obs, pos, mask, e = _inputs()
    with torch.no_grad():
        a, lp, ma, _ = actor.get_action(obs, pos, mask, deterministic=True, cond=e)
    assert a.shape == (2, 4, 1) and lp.shape == (2, 1)
