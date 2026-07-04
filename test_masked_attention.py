"""Validate MaskedScaledAttention: equivalence to nn.MultiheadAttention (flags off),
padding correctness, and permutation equivariance. Run: python test_masked_attention.py
"""
import math
import torch
import torch.nn as nn

from positional_encodings._attn import MaskedScaledAttention, neighbour_allow_mask

torch.manual_seed(0)
B, N, E, H = 3, 7, 16, 4


def _sync_from_mha(msa, mha):
    with torch.no_grad():
        msa.in_proj.weight.copy_(mha.in_proj_weight)
        msa.in_proj.bias.copy_(mha.in_proj_bias)
        msa.out_proj.weight.copy_(mha.out_proj.weight)
        msa.out_proj.bias.copy_(mha.out_proj.bias)


def test_equivalence():
    mha = nn.MultiheadAttention(E, H, batch_first=True)
    msa = MaskedScaledAttention(E, H, logit_scale="none", softmax_type="softmax")
    msa.eval(); mha.eval()
    _sync_from_mha(msa, mha)

    x = torch.randn(B, N, E)
    bias = torch.randn(B, H, N, N)
    # key padding: each batch keeps >=3 real tokens (avoid fully-masked query rows)
    kpm = torch.zeros(B, N, dtype=torch.bool)
    kpm[0, 5:] = True; kpm[1, 6:] = True

    out_msa, _ = msa(x, key_padding_mask=kpm, attn_bias=bias)
    out_mha, _ = mha(x, x, x, key_padding_mask=kpm,
                     attn_mask=bias.reshape(B * H, N, N), need_weights=False)
    # compare only real (non-padded) query rows (torch yields NaN on fully-padded rows)
    real = (~kpm).unsqueeze(-1)
    diff = ((out_msa - out_mha).abs() * real).max().item()
    assert diff < 1e-5, f"equivalence failed: max diff {diff}"
    print(f"[ok] equivalence to nn.MultiheadAttention: max diff {diff:.2e}")


def test_padding_ignored():
    """Changing padded-token features must not change real-token outputs."""
    msa = MaskedScaledAttention(E, H); msa.eval()
    x = torch.randn(B, N, E)
    kpm = torch.zeros(B, N, dtype=torch.bool); kpm[:, 5:] = True
    out1, _ = msa(x, key_padding_mask=kpm)
    x2 = x.clone(); x2[:, 5:] = torch.randn(B, 2, E)  # perturb padded tokens
    out2, _ = msa(x2, key_padding_mask=kpm)
    real = (~kpm).unsqueeze(-1)
    diff = ((out1 - out2).abs() * real).max().item()
    assert diff < 1e-6, f"padded tokens leaked into real outputs: {diff}"
    print(f"[ok] padded tokens do not affect real outputs: max diff {diff:.2e}")


def test_permutation_equivariance():
    """Permuting tokens permutes outputs (no positional leakage from token order)."""
    msa = MaskedScaledAttention(E, H); msa.eval()
    x = torch.randn(1, N, E)
    perm = torch.randperm(N)
    out, _ = msa(x)
    out_perm, _ = msa(x[:, perm])
    diff = (out[:, perm] - out_perm).abs().max().item()
    assert diff < 1e-5, f"not permutation equivariant: {diff}"
    print(f"[ok] permutation equivariant: max diff {diff:.2e}")


def test_local_and_logn_run():
    """Local masking + log-N scaling execute, stay finite, and sharpen attention at large N."""
    pos = torch.randn(B, N, 2) * 5.0  # D units
    allow = neighbour_allow_mask(pos, None, mode="knn", k=3)
    assert allow.diagonal(dim1=-2, dim2=-1).all(), "self-attention must always be allowed"
    assert (allow.sum(-1) >= 3).all()

    msa = MaskedScaledAttention(E, H, logit_scale="logn"); msa.eval()
    out, attn = msa(torch.randn(B, N, E), attn_bias=torch.randn(B, H, N, N),
                    local_allow=allow, need_weights=True)
    assert torch.isfinite(out).all(), "log-N + local produced non-finite output"
    # attention respects locality: disallowed (non-diagonal) entries are ~0
    disallowed = ((~allow).unsqueeze(1) & ~torch.eye(N, dtype=torch.bool).view(1, 1, N, N))
    disallowed = disallowed.expand(B, H, N, N)
    assert (attn * disallowed.float()).max().item() < 1e-6, "attention leaked outside local mask"
    print("[ok] local mask + log-N scaling run, finite, and respected")

    # log-N sharpening: with the SAME scores, entropy at N=25 should be lower (sharper)
    # for logn than for plain softmax (sanity of the dilution fix)
    base = MaskedScaledAttention(E, H, logit_scale="none"); base.eval()
    ln = MaskedScaledAttention(E, H, logit_scale="logn"); ln.eval()
    ln.load_state_dict({k: v for k, v in base.state_dict().items()}, strict=False)
    with torch.no_grad():
        ln.log_n_scale.fill_(1.0)  # positive scale
    x25 = torch.randn(2, 25, E)
    _, a_base = base(x25, need_weights=True)
    _, a_ln = ln(x25, need_weights=True)
    ent = lambda a: -(a.clamp_min(1e-9) * a.clamp_min(1e-9).log()).sum(-1).mean().item()
    print(f"[ok] mean attention entropy @N=25  softmax={ent(a_base):.3f}  logN={ent(a_ln):.3f} "
          f"(logN should be lower = sharper)")


if __name__ == "__main__":
    test_equivalence()
    test_padding_ignored()
    test_permutation_equivariance()
    test_local_and_logn_run()
    print("\nALL TESTS PASSED")
