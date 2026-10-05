"""Seeded construction + checksum of the cond-OFF actor / critics. Run once on
main BEFORE the Stage-10 edits to write tests/snapshots/cond_off_main.json;
tests/test_cond_bit_identity.py recomputes and compares."""
import hashlib
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import Args  # noqa: E402
from networks import TransformerActor, TransformerCritic, TransformerTQCSharedCritic  # noqa: E402

SNAP = Path(__file__).resolve().parent / "snapshots" / "cond_off_main.json"
OBS_DIM = 60


def build(seed=0):
    args = Args()
    torch.manual_seed(seed)
    actor = TransformerActor(obs_dim_per_turbine=OBS_DIM, action_dim_per_turbine=1, args=args)
    torch.manual_seed(seed + 1)
    critic = TransformerCritic(obs_dim_per_turbine=OBS_DIM, action_dim_per_turbine=1, args=args)
    torch.manual_seed(seed + 2)
    tqc = TransformerTQCSharedCritic(n_critics=2, n_quantiles=5, obs_dim_per_turbine=OBS_DIM,
                                     action_dim_per_turbine=1, args=args)
    return actor, critic, tqc


def digest(module):
    sd = module.state_dict()
    h = hashlib.sha256()
    for k in sorted(sd):
        h.update(k.encode())
        h.update(sd[k].detach().cpu().contiguous().numpy().tobytes())
    return {"keys": sorted(sd), "sha256": h.hexdigest()}


def snapshot():
    actor, critic, tqc = build()
    torch.manual_seed(123)
    obs = torch.randn(2, 4, OBS_DIM); pos = torch.randn(2, 4, 2); act = torch.randn(2, 4, 1)
    mask = torch.tensor([[False, False, False, True], [False, False, True, True]])
    with torch.no_grad():
        mean, log_std, _ = actor(obs, pos, mask)
        q = critic(obs, act, pos, mask)
        qt = tqc(obs, act, pos, mask)
    return {
        "actor": digest(actor), "critic": digest(critic), "tqc_shared": digest(tqc),
        "forward": {"mean_sum": float(mean.double().sum()), "log_std_sum": float(log_std.double().sum()),
                    "q_sum": float(q.double().sum()), "qt_sum": float(qt.double().sum())},
    }


if __name__ == "__main__":
    SNAP.write_text(json.dumps(snapshot(), indent=1))
    print("wrote", SNAP)
