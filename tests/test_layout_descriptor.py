"""Lane descriptor (helpers/layout_descriptor.py): hand-computable cases and network wiring.

Feature layout of one turbine: (side, band, lateral bin), side 0 = upstream neighbours,
side 1 = downstream. With the ``lanes2_wide_half`` preset the bands are "neighbours within
3, 5, 7, 10, 15 D" and the lateral bins sit at -6, -3, -1, 0, 1, 3, 6 D (centre: index 3).
Positions are in D, wind frame, +x downstream.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import Args  # noqa: E402
from helpers.layout_descriptor import (  # noqa: E402
    PRESETS,
    LaneDescriptorEncoder,
    check_layout_descriptor_args,
    lane_descriptor,
)
from networks import TransformerActor, TransformerCritic  # noqa: E402

WIDE = PRESETS["lanes2_wide_half"]
UP, DOWN, CENTRE = 0, 1, 3
OBS = 12


def desc(pts, mask=None, preset=WIDE, **kw):
    """Descriptor of one layout [(x, y), ...] as (turbine, side, band, lateral bin)."""
    pos = torch.tensor(pts, dtype=torch.float32)[None]
    m = None if mask is None else torch.tensor(mask)[None]
    out = lane_descriptor(pos, m, **{**preset, **kw})[0]
    return out.reshape(len(pts), 2, len(preset["nodes"]), len(preset["lat_nodes"]))


def row(n, spacing=5.0):
    return [(i * spacing, 0.0) for i in range(n)]


def test_shape_and_isolated_turbine_is_zero():
    out = lane_descriptor(torch.zeros(2, 1, 2), None, **WIDE)
    assert out.shape == (2, 1, 2 * 5 * 7) and out.dtype == torch.float32
    assert torch.count_nonzero(out) == 0


def test_one_neighbour_5D_downstream_fills_the_within_5_to_15_cells():
    d = desc([(0.0, 0.0), (5.0, 0.0)])
    want = torch.zeros(2, 5, 7)
    want[DOWN, 1:, CENTRE] = 1.0  # within 5, 7, 10, 15 D; not within 3 D
    torch.testing.assert_close(d[0], want)
    # the neighbour sees the same thing upstream
    torch.testing.assert_close(d[1], want.flip(0))


def test_half_a_diameter_to_the_left_splits_half_and_half():
    d = desc([(0.0, 0.0), (5.0, 0.5)])
    want = torch.zeros(2, 5, 7)
    want[DOWN, 1:, CENTRE] = 0.5
    want[DOWN, 1:, CENTRE + 1] = 0.5  # the bin at +1 D
    torch.testing.assert_close(d[0], want)


def test_rows_beyond_3D_count_half():
    d = desc([(0.0, 0.0), (5.0, 6.0)])
    want = torch.zeros(2, 5, 7)
    want[DOWN, 1:, 6] = 0.5
    torch.testing.assert_close(d[0], want)


def test_neighbour_fades_in_between_half_and_one_and_a_half_D():
    assert torch.count_nonzero(desc([(0.0, 0.0), (0.5, 0.0)])[0]) == 0
    d = desc([(0.0, 0.0), (1.0, 0.0)])[0]
    torch.testing.assert_close(d[DOWN, :, CENTRE], torch.full((5,), 0.5))


def test_second_of_a_long_row_differs_from_the_middle_of_three_in_two_cells():
    long, short = desc(row(100))[1], desc(row(3))[1]
    diff = long - short
    want = torch.zeros(2, 5, 7)
    want[DOWN, 3:, CENTRE] = 1.0  # the second downstream neighbour, 10 D: within 10 and 15 D
    torch.testing.assert_close(diff, want, rtol=0, atol=1e-6)


def test_deep_in_a_long_row_equals_the_interior_of_a_row_of_five():
    torch.testing.assert_close(desc(row(100))[50], desc(row(5))[2], rtol=0, atol=1e-6)
    torch.testing.assert_close(desc(row(100))[0], desc(row(3))[0], rtol=0, atol=1e-6)
    torch.testing.assert_close(desc(row(100))[99], desc(row(3))[2], rtol=0, atol=1e-6)


def test_first_none_counts_every_neighbour_in_reach():
    d = desc(row(100), first=None)[50]
    # neighbours at 5, 10, 15, 20 D (the last one faded to 2/7): within 15 D holds 3 + 2/7
    assert d[DOWN, 4, CENTRE] == pytest.approx(3 + 2 / 7, abs=1e-5)
    assert d[DOWN, 1, CENTRE] == pytest.approx(1.0, abs=1e-6)


def test_padded_turbines_change_nothing_and_get_zeros():
    pts = [(0.0, 0.0), (5.0, 0.5), (11.0, -2.0), (4.0, 7.0)]
    base = desc(pts)
    padded = desc(pts + [(2.5, 0.0), (0.0, 0.0), (7.0, 0.2)],
                  mask=[False] * 4 + [True] * 3)
    torch.testing.assert_close(padded[:4], base)
    assert torch.count_nonzero(padded[4:]) == 0


def test_turbine_beyond_the_reach_changes_nothing():
    pts = [(0.0, 0.0), (5.0, 0.5), (11.0, -2.0), (4.0, 7.0)]
    base = desc(pts)
    for far in [(40.0, 0.0), (-25.0, 1.0), (5.0, 21.0), (16.0, -12.0)]:
        torch.testing.assert_close(desc(pts + [far])[:4], base)


def test_permuting_turbines_permutes_the_descriptor():
    g = torch.Generator().manual_seed(0)
    pos = torch.rand(3, 9, 2, generator=g) * 25.0
    perm = torch.randperm(9, generator=g)
    for name in PRESETS:
        a = lane_descriptor(pos, None, **PRESETS[name])
        b = lane_descriptor(pos[:, perm], None, **PRESETS[name])
        torch.testing.assert_close(b, a[:, perm], rtol=0, atol=1e-6)


def test_batch_rows_are_independent_and_chunking_does_not_matter():
    g = torch.Generator().manual_seed(1)
    pos = torch.rand(4, 11, 2, generator=g) * 20.0
    mask = torch.zeros(4, 11, dtype=torch.bool)
    mask[1, 8:] = True
    full = lane_descriptor(pos, mask, **WIDE)
    for b in range(4):
        torch.testing.assert_close(lane_descriptor(pos[b:b + 1], mask[b:b + 1], **WIDE), full[b:b + 1])
    torch.testing.assert_close(lane_descriptor(pos, mask, chunk_elems=1, **WIDE), full, rtol=0, atol=1e-6)


def test_translation_leaves_the_descriptor_unchanged():
    g = torch.Generator().manual_seed(2)
    pos = torch.rand(1, 8, 2, generator=g) * 20.0
    shift = torch.tensor([7.3, -3.1])
    torch.testing.assert_close(lane_descriptor(pos + shift, None, **WIDE),
                               lane_descriptor(pos, None, **WIDE), rtol=0, atol=1e-4)


def test_encoder_presets_overrides_and_no_gradient_to_positions():
    enc = LaneDescriptorEncoder(16, "lanes2_wide_half")
    assert enc.feature_dim == 70
    assert LaneDescriptorEncoder(16, "lanes2").feature_dim == 50
    assert LaneDescriptorEncoder(16, "lanes2", lat_nodes=[-1, 0, 1], lat_outer=2.0).feature_dim == 30
    pos = (torch.rand(2, 5, 2) * 20.0).requires_grad_()
    out = enc(pos, None)
    assert out.shape == (2, 5, 16)
    out.sum().backward()
    assert pos.grad is None
    with pytest.raises(ValueError, match="layout_descriptor"):
        LaneDescriptorEncoder(16, "nope")
    with pytest.raises(TypeError):
        LaneDescriptorEncoder(16, "lanes2", bands=3)


# ---------------------------------------------------------------- networks --
def _inputs():
    g = torch.Generator().manual_seed(3)
    obs = torch.randn(2, 4, OBS, generator=g)
    act = torch.randn(2, 4, 1, generator=g)
    pos = torch.rand(2, 4, 2, generator=g) * 12.0
    mask = torch.tensor([[False, False, False, True], [False, False, True, True]])
    return obs, act, pos, mask


def _build(cls, args, seed=0):
    torch.manual_seed(seed)
    return cls(obs_dim_per_turbine=OBS, embed_dim=32, pos_embed_dim=8, num_heads=2,
               num_layers=1, args=args)


def _run(net, obs, act, pos, mask):
    if isinstance(net, TransformerActor):
        return net(obs, pos, mask)[0]
    return net(obs, act, pos, mask)


@pytest.mark.parametrize("cls", [TransformerActor, TransformerCritic])
@pytest.mark.parametrize("mode", ["concat", "add"])
def test_networks_build_and_use_the_descriptor(cls, mode):
    obs, act, pos, mask = _inputs()
    off = _build(cls, Args())
    on = _build(cls, Args(layout_descriptor="lanes2_wide_half", layout_descriptor_mode=mode))
    new = set(on.state_dict()) - set(off.state_dict())
    assert set(off.state_dict()) <= set(on.state_dict())
    assert new and all(k.startswith("layout_desc_") for k in new)
    assert any(k.startswith("layout_desc_proj") for k in new) == (mode == "concat")
    out = _run(on, obs, act, pos, mask)
    assert torch.isfinite(out).all()
    out.sum().backward()
    assert on.layout_desc_encoder.proj[0].weight.grad.abs().sum() > 0


@pytest.mark.parametrize("cls", [TransformerActor, TransformerCritic])
def test_descriptor_reaches_the_output(cls):
    """With no positional encoding the output can only depend on the layout through the
    descriptor: off, moving a neighbour changes nothing; on, it does."""
    obs, act, pos, mask = _inputs()
    moved = pos.clone()
    moved[0, 1] = moved[0, 0] + torch.tensor([5.0, 0.0])
    for name, changes in ((None, False), ("lanes2", True)):
        net = _build(cls, Args(layout_descriptor=name), seed=1)
        net.pos_encoder, net.rel_pos_bias = None, None
        net.input_proj = torch.nn.Identity()
        a, b = _run(net, obs, act, pos, mask), _run(net, obs, act, moved, mask)
        assert (not torch.allclose(a, b)) == changes


@pytest.mark.parametrize("cls", [TransformerActor, TransformerCritic])
def test_flag_off_and_args_without_the_fields_are_the_same_network(cls):
    """An args object saved before the flags existed builds the network of today's default."""
    obs, act, pos, mask = _inputs()
    old_args = SimpleNamespace(**{k: v for k, v in vars(Args()).items()
                                  if not k.startswith("layout_descriptor")})
    assert not hasattr(old_args, "layout_descriptor")
    new, old = _build(cls, Args()), _build(cls, old_args)
    assert list(new.state_dict()) == list(old.state_dict())
    assert not any("layout_desc" in k for k in new.state_dict())
    torch.testing.assert_close(_run(new, obs, act, pos, mask), _run(old, obs, act, pos, mask))


def test_kwargs_flag_overrides_the_preset():
    args = Args(layout_descriptor="lanes2", layout_descriptor_kwargs='{"first": null, "outer": 30.0}')
    net = _build(TransformerActor, args)
    assert net.layout_desc_encoder.params["first"] is None
    assert net.layout_desc_encoder.params["outer"] == 30.0
    assert net.layout_desc_encoder.params["nodes"] == PRESETS["lanes2"]["nodes"]


def test_trainer_check_refuses_wrong_frame_compile_and_unknown_names():
    check_layout_descriptor_args(Args())
    check_layout_descriptor_args(Args(use_wind_relative_pos=False, compile=True))  # flag off
    check_layout_descriptor_args(Args(layout_descriptor="lanes2"))
    with pytest.raises(ValueError, match="use_wind_relative_pos"):
        check_layout_descriptor_args(Args(layout_descriptor="lanes2", use_wind_relative_pos=False))
    with pytest.raises(ValueError, match="compile"):
        check_layout_descriptor_args(Args(layout_descriptor="lanes2", compile=True))
    with pytest.raises(ValueError, match="Unknown"):
        check_layout_descriptor_args(Args(layout_descriptor="roses"))
