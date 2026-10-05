"""WindFarmAgent / BatchPreparer thread params + action_hist into the actor (and phi)."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import Args  # noqa: E402
from helpers.agent import BatchPreparer, InferenceBatch, WindFarmAgent  # noqa: E402
from helpers.param_cond import ParamNormalizer  # noqa: E402
from networks import AdaptationModule, TransformerActor  # noqa: E402

OBS, N, T = 60, 2, 4


class _FakeVec:
    """Mimics envs.env.get_attr of an AsyncVectorEnv over MultiLayoutEnv."""
    def __init__(self):
        self.tok = np.zeros((T, 6)); self.tok[:, 1] = 1.0; self.tok[0, 0] = 0.6  # yaw_exp hi on token 0
        self.farm = {"k1": 0.09, "k2": 0.008, "d_particle": 0.65, "mann_L": 69.0, "mann_GAMMA": 3.95, "mann_AE": 0.0097}
        self.calls = []
    def get_attr(self, name):
        self.calls.append(name)
        vals = {
            "wd": [270.0] * N, "turbine_positions": [np.zeros((T, 2), np.float32)] * N,
            "attention_mask": [np.array([False, False, False, True])] * N,
            "turbine_params_tok": [self.tok] * N, "farm_params": [self.farm] * N,
            "action_history": [np.full((T, 15), 0.25, np.float32)] * N,
        }
        return vals[name]


def _norm():
    return ParamNormalizer({"yaw_exp": (-0.4, 0.6)}, {k: v for k, v in _FakeVec().farm.items()},
                           {k: 1.0 for k in _FakeVec().farm}, source="turbine_farm")


def test_inference_batch_has_cond_fields():
    b = InferenceBatch(obs=torch.zeros(1), positions=torch.zeros(1), mask=torch.zeros(1))
    assert b.params is None and b.action_hist is None


def test_preparer_resolves_params_and_action_hist_from_envs():
    prep = BatchPreparer(torch.device("cpu"), rotor_diameter=100.0, use_wind_relative=False,
                         cond_normalizer=_norm(), use_action_hist=True)
    envs = SimpleNamespace(env=_FakeVec())
    batch = prep.from_envs(envs, np.zeros((N, T, OBS), np.float32))
    assert batch.params.shape == (N, T, 12) and batch.params.dtype == torch.float32
    assert batch.params[0, 0, 0].item() == pytest.approx(1.0)      # yaw_exp hi -> +1
    assert torch.all(batch.params[:, 3] == 0)                       # padded token
    assert batch.action_hist.shape == (N, T, 15) and batch.action_hist[0, 0, 0].item() == 0.25
    # precomputed values are used as-is (no IPC)
    envs.env.calls.clear()
    e = np.ones((N, T, 12), np.float32); ah = np.zeros((N, T, 15), np.float32)
    b2 = prep.from_envs(envs, np.zeros((N, T, OBS), np.float32), wind_dirs=np.zeros(N), raw_positions=np.zeros((N, T, 2)),
                        masks=np.zeros((N, T), bool), params=e, action_hist=ah)
    assert envs.env.calls == [] and torch.all(b2.params == 1)


def test_preparer_without_cond_leaves_fields_none():
    prep = BatchPreparer(torch.device("cpu"), rotor_diameter=100.0)
    batch = prep.from_envs(SimpleNamespace(env=_FakeVec()), np.zeros((N, T, OBS), np.float32))
    assert batch.params is None and batch.action_hist is None


def test_agent_act_passes_cond_to_cond_actor_and_none_otherwise():
    envs = SimpleNamespace(env=_FakeVec())
    obs = np.zeros((N, T, OBS), np.float32)
    plain = WindFarmAgent(TransformerActor(obs_dim_per_turbine=OBS, args=Args()), torch.device("cpu"), 100.0)
    assert plain.act(envs, obs, deterministic=True).shape == (N, T)
    actor = TransformerActor(obs_dim_per_turbine=OBS, args=Args(cond_source="turbine_farm", cond_latent_dim=8))
    agent = WindFarmAgent(actor, torch.device("cpu"), 100.0, cond_normalizer=_norm())
    a1 = agent.act(envs, obs, deterministic=True)
    assert a1.shape == (N, T)
    # a different e changes the action
    e2 = np.full((N, T, 12), -1.0, np.float32)
    a2 = agent.act(envs, obs, deterministic=True, params=e2)
    assert not np.allclose(a1, a2)
    # cond actor without a normaliser and without params is a hard error
    with pytest.raises(ValueError, match="cond"):
        WindFarmAgent(actor, torch.device("cpu"), 100.0).act(envs, obs)


def test_agent_with_adaptation_uses_z_hat():
    envs = SimpleNamespace(env=_FakeVec())
    obs = np.zeros((N, T, OBS), np.float32)
    args = Args(cond_source="turbine_farm", cond_latent_dim=8)
    actor = TransformerActor(obs_dim_per_turbine=OBS, args=args)
    phi = AdaptationModule(OBS, 15, out_dim=actor.cond_dim, args=args)
    agent = WindFarmAgent(actor, torch.device("cpu"), 100.0, use_action_hist=True, adaptation=phi)
    a_hat = agent.act(envs, obs, deterministic=True)          # no params needed: z_hat from phi
    assert a_hat.shape == (N, T)
    # equals the actor fed phi's latent directly
    prep = agent.batch_preparer
    b = prep.from_envs(envs, obs)
    with torch.no_grad():
        z = phi(b.obs, b.action_hist, b.positions, b.mask)
        ref = actor.get_action(b.obs, b.positions, b.mask, deterministic=True, cond=z, cond_is_latent=True)[0]
    assert np.allclose(a_hat, ref.squeeze(-1).numpy(), atol=1e-6)
