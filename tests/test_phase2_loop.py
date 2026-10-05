"""run_phase2's DAgger loop on a fake SAME_STEP vector env: per-round save callback,
stop callback, held-out metric, and the guarded wrapper."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import Args  # noqa: E402
from helpers.param_cond import ParamNormalizer  # noqa: E402
from helpers.phase2_adapt import run_phase2, run_phase2_guarded  # noqa: E402
from networks import TransformerActor  # noqa: E402

OBS, T, N = 20, 4, 2


class _Inner:
    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)
        self.t = 0
        self.tok = np.zeros((T, 6)); self.tok[:, 1] = 1.0
        self.farm = {"k1": 0.09, "k2": 0.008, "d_particle": 0.65, "mann_L": 69.0, "mann_GAMMA": 3.95, "mann_AE": 0.0097}
        self.reset()

    def reset(self):
        self.tok[:3, 0] = self.rng.uniform(-0.4, 0.6, size=3)
        self.hist = np.zeros((T, 15), np.float32)
        self.t = 0


class _FakeVec:
    """Mimics RecordEpisodeVals(AsyncVectorEnv(MultiLayoutEnv)) for the phase-2 loop."""

    def __init__(self, ep_len=5):
        self.inner = [_Inner(i) for i in range(N)]
        self.ep_len = ep_len
        self.env = self                       # trainer-style envs.env.get_attr

    def get_attr(self, name):
        vals = {
            "wd": lambda e: 270.0, "turbine_positions": lambda e: np.zeros((T, 2), np.float32),
            "attention_mask": lambda e: np.array([False, False, False, True]),
            "turbine_params_tok": lambda e: e.tok.copy(), "farm_params": lambda e: dict(e.farm),
            "action_history": lambda e: e.hist.copy(),
        }
        return [vals[name](e) for e in self.inner]

    def reset(self, seed=None):
        for e in self.inner:
            e.reset()
        return np.zeros((N, T, OBS), np.float32), {}

    def step(self, actions):
        dones = np.zeros(N, bool)
        for i, e in enumerate(self.inner):
            e.hist[:, :-1] = e.hist[:, 1:]; e.hist[:, -1] = np.asarray(actions[i])[:T]
            e.t += 1
            if e.t >= self.ep_len:          # SAME_STEP: reset inside the step
                dones[i] = True; e.reset()
        return np.zeros((N, T, OBS), np.float32), np.zeros(N), dones, np.zeros(N, bool), {}


def _setup(rounds=2, steps=6):
    args = Args(cond_source="turbine_farm", cond_latent_dim=8, cond_action_hist=True, adapt_rounds=rounds,
                adapt_round_steps=steps, adapt_warm_steps=0, adapt_fit_steps=3, batch_size=4, num_envs=N,
                turb_dr=("yaw_exp=-0.4:0.6",))
    actor = TransformerActor(obs_dim_per_turbine=OBS, args=args)
    norm = ParamNormalizer({"yaw_exp": (-0.4, 0.6)}, _Inner(0).farm, {k: 1.0 for k in _Inner(0).farm})
    kw = dict(trunk_kwargs={"args": args}, rotor_diameter=100.0, use_profiles=False, obs_normalizer=None,
              wd_attr="wd", profile_registry=None, n_turbines_max=T, obs_dim=OBS, action_dim=1, global_step=0)
    return args, actor, norm, kw


def test_saves_after_every_round_and_reports_heldout():
    args, actor, norm, kw = _setup(rounds=2, steps=6)
    saved = []
    out = run_phase2(args, actor, _FakeVec(), None, norm, torch.device("cpu"),
                     save_fn=lambda sd, metrics: saved.append((set(sd), len(metrics["rounds"]))), **kw)
    assert [n for _, n in saved] == [1, 2]                       # one save per round
    assert all("head.weight" in k for k, _ in saved)
    assert len(out["metrics"]["rounds"]) == 2
    r = out["metrics"]["rounds"][-1]
    assert r["n_pairs"] == 2 * N * 6 and "heldout_mse" in r and "mse" in r
    assert np.isfinite(r["heldout_mse"])


def test_stop_callback_ends_after_current_round():
    args, actor, norm, kw = _setup(rounds=3, steps=4)
    calls = {"n": 0}
    def stop():
        calls["n"] += 1
        return calls["n"] >= 1                                    # stop requested after round 1
    out = run_phase2(args, actor, _FakeVec(), None, norm, torch.device("cpu"), stop_fn=stop, **kw)
    assert len(out["metrics"]["rounds"]) == 1 and out["metrics"]["stopped_early"] is True
    assert out["adapt_state_dict"] is not None


def test_guarded_wrapper_swallows_and_reports():
    def boom(**_):
        raise RuntimeError("phi exploded")
    res = run_phase2_guarded(boom, x=1)
    assert res is None
    ok = run_phase2_guarded(lambda **kw: {"adapt_state_dict": {}}, x=1)
    assert ok == {"adapt_state_dict": {}}
