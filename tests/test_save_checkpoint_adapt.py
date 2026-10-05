"""save_checkpoint carries adapt_state_dict / student_actor_state_dict (Stage 10 phase 2)."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import Args  # noqa: E402
from helpers.helper_funcs import save_checkpoint  # noqa: E402
from networks import TransformerActor  # noqa: E402


def test_adapt_state_dict_saved_and_absent_by_default(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    args = Args(cond_source="turbine_farm", cond_latent_dim=8)
    actor = TransformerActor(obs_dim_per_turbine=60, args=args)
    opt = torch.optim.Adam(actor.parameters())
    p = save_checkpoint(actor, None, None, opt, opt, 0, "r", args, tqc_critic=actor)
    ck = torch.load(p, weights_only=False)
    assert "adapt_state_dict" not in ck and "student_actor_state_dict" not in ck
    assert ck["args"]["cond_source"] == "turbine_farm"
    p = save_checkpoint(actor, None, None, opt, opt, 1, "r", args, tqc_critic=actor,
                        adapt_state_dict={"head.weight": torch.zeros(2)},
                        student_actor_state_dict={"x": torch.ones(1)})
    ck = torch.load(p, weights_only=False)
    assert torch.equal(ck["adapt_state_dict"]["head.weight"], torch.zeros(2))
    assert torch.equal(ck["student_actor_state_dict"]["x"], torch.ones(1))
