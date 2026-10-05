"""Phase-2 adaptation fit (helpers/phase2_adapt.py): pure-torch pieces.
The DAgger rollout loop is exercised by the CPU smoke (tests/smoke_cond.sh)."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import Args  # noqa: E402
from helpers.phase2_adapt import (  # noqa: E402
    ArrayBatchSource,
    fit_adaptation,
    linear_probe_r2,
    make_target_z,
    per_dim_rel_error,
)
from networks import AdaptationModule, TransformerActor  # noqa: E402

OBS, AH, T, B = 20, 15, 4, 32


def _synthetic(n=256, seed=0, dz=8):
    """obs carries e linearly in its first 12 columns (identifiable); action_hist = noise."""
    g = torch.Generator().manual_seed(seed)
    e = torch.randn(n, T, 12, generator=g)
    obs = torch.randn(n, T, OBS, generator=g) * 0.1
    obs[..., :12] += e
    ah = torch.randn(n, T, AH, generator=g)
    pos = torch.randn(n, T, 2, generator=g)
    mask = torch.zeros(n, T, dtype=torch.bool); mask[:, -1] = True
    e[mask] = 0.0
    return {"observations": obs, "action_hist": ah, "positions": pos, "attention_mask": mask, "params": e}


def test_make_target_z_follows_actor_encoder():
    e = torch.randn(2, T, 12)
    a8 = TransformerActor(obs_dim_per_turbine=OBS, args=Args(cond_source="turbine_farm", cond_latent_dim=8))
    a0 = TransformerActor(obs_dim_per_turbine=OBS, args=Args(cond_source="turbine_farm", cond_latent_dim=0))
    with torch.no_grad():
        assert torch.equal(make_target_z(a8, e), a8.param_encoder(e))
        assert torch.equal(make_target_z(a0, e), e)
        assert make_target_z(a8, e).requires_grad is False


def test_fit_decreases_masked_latent_mse_and_ignores_padding():
    torch.manual_seed(0)
    args = Args(cond_source="turbine_farm", cond_latent_dim=0, cond_mode="concat")
    actor = TransformerActor(obs_dim_per_turbine=OBS, args=args)
    phi = AdaptationModule(OBS, AH, out_dim=actor.cond_dim, args=args)
    data = _synthetic()
    src = ArrayBatchSource(data, batch_size=B, device=torch.device("cpu"), seed=0)
    opt = torch.optim.Adam(phi.parameters(), lr=1e-3)
    hist = fit_adaptation(phi, actor, src, steps=150, optimizer=opt, loss_kind="latent",
                          device=torch.device("cpu"), log_every=50)
    assert len(hist) == 3 and hist[-1]["mse"] < 0.5 * hist[0]["mse"]
    # padded tokens never enter the loss: perturbing their targets changes nothing
    batch = src.sample()
    with torch.no_grad():
        z_hat = phi(batch["observations"], batch["action_hist"], batch["positions"], batch["attention_mask"])
    from networks import masked_mse
    tgt = make_target_z(actor, batch["params"])
    l1 = masked_mse(z_hat, tgt, batch["attention_mask"])
    tgt2 = tgt.clone(); tgt2[batch["attention_mask"]] += 100.0
    assert torch.allclose(l1, masked_mse(z_hat, tgt2, batch["attention_mask"]))


def test_fit_latent_action_trains_student_and_leaves_teacher_frozen():
    torch.manual_seed(1)
    args = Args(cond_source="turbine_farm", cond_latent_dim=8, cond_mode="concat")
    teacher = TransformerActor(obs_dim_per_turbine=OBS, args=args)
    import copy
    student = copy.deepcopy(teacher)
    phi = AdaptationModule(OBS, AH, out_dim=teacher.cond_dim, args=args)
    data = _synthetic(dz=8)
    src = ArrayBatchSource(data, batch_size=B, device=torch.device("cpu"), seed=1)
    opt = torch.optim.Adam(list(phi.parameters()) + list(student.parameters()), lr=1e-3)
    t_before = {k: v.clone() for k, v in teacher.state_dict().items()}
    s_before = {k: v.clone() for k, v in student.state_dict().items()}
    hist = fit_adaptation(phi, teacher, src, steps=20, optimizer=opt, loss_kind="latent_action",
                          device=torch.device("cpu"), student=student, log_every=10)
    assert all(torch.equal(t_before[k], v) for k, v in teacher.state_dict().items())
    assert any(not torch.equal(s_before[k], v) for k, v in student.state_dict().items())
    assert "action_mse" in hist[-1] and hist[-1]["action_mse"] >= 0


def test_linear_probe_r2_and_rel_error():
    g = torch.Generator().manual_seed(2)
    e = torch.randn(500, T, 12, generator=g)
    mask = torch.zeros(500, T, dtype=torch.bool); mask[:, -1] = True
    W = torch.randn(12, 8, generator=g)
    z = e @ W                                   # e fully recoverable from z (rank 8 -> only 8 dims!)
    r2 = linear_probe_r2(z, e, mask)
    assert r2.shape == (12,) and np.all(r2 <= 1.0 + 1e-6)
    z_full = torch.cat([e, torch.zeros(500, T, 1)], -1)     # identity -> R^2 = 1 on every key
    r2_full = linear_probe_r2(z_full, e, mask)
    assert np.allclose(r2_full, 1.0, atol=1e-4)
    rel = per_dim_rel_error(z_full[..., :12], e, mask)
    assert rel.shape == (12,) and np.allclose(rel, 0.0, atol=1e-6)
