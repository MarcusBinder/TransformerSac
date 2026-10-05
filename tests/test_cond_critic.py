"""Asymmetric critic: raw e concatenated per token (cond_critic raw) or ignored (none)."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import Args  # noqa: E402
from networks import TransformerCritic, TransformerTQCCritic, TransformerTQCSharedCritic  # noqa: E402

OBS = 60


def _inputs():
    g = torch.Generator().manual_seed(3)
    obs = torch.randn(2, 4, OBS, generator=g); act = torch.randn(2, 4, 1, generator=g)
    pos = torch.randn(2, 4, 2, generator=g); e = torch.randn(2, 4, 12, generator=g)
    mask = torch.tensor([[False, False, False, True], [False, False, True, True]])
    return obs, act, pos, mask, e


@pytest.mark.parametrize("cls,kw", [
    (TransformerCritic, {}),
    (TransformerTQCCritic, {"n_critics": 2, "n_quantiles": 5}),
    (TransformerTQCSharedCritic, {"n_critics": 2, "n_quantiles": 5}),
])
def test_raw_cond_widens_encoder_and_is_required(cls, kw):
    args = Args(cond_source="turbine_farm", cond_latent_dim=8, cond_critic="raw")
    c = cls(obs_dim_per_turbine=OBS, action_dim_per_turbine=1, args=args, **kw)
    trunk = c if cls is TransformerCritic else (c.trunk if hasattr(c, "trunk") else c.critics[0])
    assert trunk.obs_action_encoder[0].weight.shape == (128, OBS + 1 + 12)
    assert trunk.cond_critic_dim == 12
    obs, act, pos, mask, e = _inputs()
    with pytest.raises(ValueError, match="cond"):
        c(obs, act, pos, mask)
    with torch.no_grad():
        q1 = c(obs, act, pos, mask, cond=e)
        q2 = c(obs, act, pos, mask, cond=e * 2)
    assert q1.shape[-1] in (1, 5)
    assert not torch.allclose(q1, q2)


def test_cond_critic_none_keeps_critic_unchanged():
    args_on = Args(cond_source="turbine_farm", cond_latent_dim=8, cond_critic="none")
    torch.manual_seed(0)
    c_on = TransformerCritic(obs_dim_per_turbine=OBS, args=args_on)
    torch.manual_seed(0)
    c_off = TransformerCritic(obs_dim_per_turbine=OBS, args=Args())
    assert c_on.cond_critic_dim == 0
    assert c_on.state_dict().keys() == c_off.state_dict().keys()
    obs, act, pos, mask, e = _inputs()
    with torch.no_grad():
        assert torch.equal(c_on(obs, act, pos, mask), c_off(obs, act, pos, mask))
        assert torch.equal(c_on(obs, act, pos, mask, cond=None), c_off(obs, act, pos, mask))
    with pytest.raises(ValueError, match="cond"):
        c_on(obs, act, pos, mask, cond=e)


def test_cond_source_none_ignores_cond_critic_raw():
    c = TransformerCritic(obs_dim_per_turbine=OBS, args=Args(cond_critic="raw"))
    assert c.cond_critic_dim == 0
