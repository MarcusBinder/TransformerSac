"""bf16 autocast keeps the cond path finite; torch.compile sees one graph with cond always a tensor."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import Args  # noqa: E402
from networks import TransformerActor  # noqa: E402

OBS = 60


def _io():
    g = torch.Generator().manual_seed(9)
    return (torch.randn(3, 5, OBS, generator=g), torch.randn(3, 5, 2, generator=g),
            torch.tensor([[False] * 5, [False] * 4 + [True], [False] * 3 + [True] * 2]),
            torch.randn(3, 5, 12, generator=g))


@pytest.mark.parametrize("mode", ["concat", "film"])
def test_bf16_autocast_finite(mode):
    actor = TransformerActor(obs_dim_per_turbine=OBS, args=Args(cond_source="turbine_farm", cond_latent_dim=8, cond_mode=mode))
    obs, pos, mask, e = _io()
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        mean, log_std, _ = actor(obs, pos, mask, cond=e)
    assert torch.isfinite(mean).all() and torch.isfinite(log_std).all()
    assert e.dtype == torch.float32


def test_compile_single_graph_with_cond():
    import torch._dynamo as dynamo
    dynamo.reset()
    actor = TransformerActor(obs_dim_per_turbine=OBS, args=Args(cond_source="turbine_farm", cond_latent_dim=8))
    obs, pos, mask, e = _io()
    counts = {"frames": 0}
    def backend(gm, example_inputs):
        counts["frames"] += 1
        return gm.forward
    fwd = torch.compile(actor.forward, backend=backend)
    with torch.no_grad():
        fwd(obs, pos, mask, cond=e)
        fwd(obs + 1, pos, mask, cond=e * 2)
    assert counts["frames"] == 1
    dynamo.reset()
