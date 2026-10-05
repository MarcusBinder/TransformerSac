"""AdaptationModule phi: actor trunk over [obs | action hist] -> z_hat (dz or 12)."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import Args  # noqa: E402
from networks import AdaptationModule, TransformerActor, masked_mse  # noqa: E402

OBS, AH = 60, 15


def test_shape_and_no_action_heads():
    args = Args(cond_source="turbine_farm", cond_latent_dim=8)
    phi = AdaptationModule(OBS, AH, out_dim=8, args=args)
    keys = list(phi.state_dict())
    assert not any(".fc_mean" in k or ".fc_logstd" in k or "param_encoder" in k or "film" in k for k in keys)
    assert phi.trunk.obs_encoder[0].weight.shape == (128, OBS + AH)
    assert phi.trunk.cond_on is False
    obs = torch.randn(2, 4, OBS); ah = torch.randn(2, 4, AH); pos = torch.randn(2, 4, 2)
    mask = torch.tensor([[False, False, False, True], [False, False, True, True]])
    z = phi(obs, ah, pos, mask)
    assert z.shape == (2, 4, 8)


def test_masked_mse_ignores_padding():
    pred = torch.zeros(2, 3, 4); tgt = torch.zeros(2, 3, 4)
    tgt[:, 2] = 100.0                       # only the padded token differs
    mask = torch.tensor([[False, False, True], [False, False, True]])
    assert masked_mse(pred, tgt, mask).item() == 0.0
    tgt[0, 0] = 1.0                         # one real token off by 1 on 4 dims
    # mean over real tokens (4) and dims (4): 4 / 16
    assert abs(masked_mse(pred, tgt, mask).item() - 4 / 16) < 1e-6


def test_encode_cond_matches_param_encoder_and_identity_for_dz0():
    e = torch.randn(2, 4, 12)
    a8 = TransformerActor(obs_dim_per_turbine=OBS, args=Args(cond_source="turbine_farm", cond_latent_dim=8))
    with torch.no_grad():
        assert torch.equal(a8.encode_cond(e), a8.param_encoder(e))
    a0 = TransformerActor(obs_dim_per_turbine=OBS, args=Args(cond_source="turbine_farm", cond_latent_dim=0))
    assert torch.equal(a0.encode_cond(e), e)


def test_adaptation_target_dim_follows_actor():
    args = Args(cond_source="turbine_farm", cond_latent_dim=0)
    actor = TransformerActor(obs_dim_per_turbine=OBS, args=args)
    phi = AdaptationModule(OBS, AH, out_dim=actor.cond_dim, args=args)
    assert phi.head.out_features == 12
