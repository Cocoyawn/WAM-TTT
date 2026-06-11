"""
Standalone unit tests for src/models/ttt.py

Scope: tests the TTT fast-weight layer *in isolation* (no dependency on
policies.py / generator.py / the VLM), so the module can be validated as a
drop-in attention replacement before any integration.

Run:
    python -m src.models.test_ttt          # from the VLANeXt repo root
    # or
    python src/models/test_ttt.py

What is checked:
  1. import + construction of both variants (no hard triton dependency)
  2. forward output shape + returned fast-weight state dict
  3. causal variant is actually causal  (perturbing token p never changes
     outputs < p  -- bitwise, since future tokens are not in their graph)
  4. bidirectional variant is actually bidirectional (perturbing token p DOES
     change earlier outputs)
  5. backward / autograd works (finite grads on input proj + fast weights)
  6. fast-weight state can be fed back in via `info` (stateful chaining)
"""

import sys
import os

import torch

# Make `torch.compile` failures fall back to eager instead of crashing the test,
# so correctness is validated even on backends where inductor is unhappy.
try:
    torch._dynamo.config.suppress_errors = True
except Exception:
    pass

# Allow running both as a module (-m src.models.test_ttt) and as a script.
try:
    from .ttt import (
        FastWeightGluMLPMultihead,
        causal_block_fast_weight_swish_glu,
        fast_weight_swish_glu_weight_norm_mini_batch_apply,
        TTTOperator,
        _FUSED_KERNELS_AVAILABLE,
    )
except ImportError:
    sys.path.insert(0, os.path.dirname(__file__))
    from ttt import (
        FastWeightGluMLPMultihead,
        causal_block_fast_weight_swish_glu,
        fast_weight_swish_glu_weight_norm_mini_batch_apply,
        TTTOperator,
        _FUSED_KERNELS_AVAILABLE,
    )


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DIM = 64
HEAD_DIM = 16          # -> num_heads = 4
B, L = 2, 32


def _bidirectional_info():
    # Full-sequence update then full-sequence apply == bidirectional.
    return {
        "ttt_op_order": [
            TTTOperator(start=0, end=-1, update=True, apply=False),
            TTTOperator(start=0, end=-1, update=False, apply=True),
        ]
    }


def _make_layer(causal, chunk_size=1):
    torch.manual_seed(0)
    layer = FastWeightGluMLPMultihead(
        dim=DIM, head_dim=HEAD_DIM, inter_multi=1,
        base_lr=0.01, muon_update_steps=0,
        causal=causal, chunk_size=chunk_size,
    ).to(DEVICE)
    layer.eval()  # deterministic; no dropout anyway
    return layer


def test_construct_and_shape():
    for causal in (False, True):
        layer = _make_layer(causal)
        x = torch.randn(B, L, DIM, device=DEVICE)
        info = {} if causal else _bidirectional_info()
        out, state = layer(x, info)
        assert out.shape == (B, L, DIM), (causal, out.shape)
        for key in ("w0", "w1", "w2"):
            assert key in state, key
            assert torch.isfinite(state[key]).all(), (causal, key)
        assert torch.isfinite(out).all(), causal
    print("[ok] construct + forward shape + finite output + state dict")


def test_no_hard_triton_dependency():
    # Module must import and run without the optional fused triton kernels.
    print(f"[ok] fused kernels available = {_FUSED_KERNELS_AVAILABLE} "
          f"(not required for default path)")
    # Requesting fused kernels when unavailable must raise clearly.
    if not _FUSED_KERNELS_AVAILABLE:
        try:
            FastWeightGluMLPMultihead(dim=DIM, head_dim=HEAD_DIM,
                                      use_fused_kernels=True).to(DEVICE)
        except ImportError:
            print("[ok] use_fused_kernels=True raises ImportError when missing")
        else:
            raise AssertionError("expected ImportError for missing fused kernels")


@torch.no_grad()
def test_causal_is_causal():
    """Perturbing token p must NOT change any output < p (strict, chunk_size=1)."""
    layer = _make_layer(causal=True, chunk_size=1)
    x = torch.randn(B, L, DIM, device=DEVICE)
    out_ref, _ = layer(x, {})

    p = L // 2
    x2 = x.clone()
    x2[:, p, :] += torch.randn(B, DIM, device=DEVICE) * 3.0  # large perturbation
    out_new, _ = layer(x2, {})

    # outputs strictly before p must be unchanged
    before = (out_ref[:, :p] - out_new[:, :p]).abs().max().item()
    # output at/after p must change (sanity: perturbation actually propagates)
    after = (out_ref[:, p:] - out_new[:, p:]).abs().max().item()

    assert before < 1e-5, f"causality violated: max diff before p = {before}"
    assert after > 1e-4, f"perturbation had no downstream effect: {after}"
    print(f"[ok] causal: diff(<p)={before:.2e} (==0)  diff(>=p)={after:.2e} (>0)")


@torch.no_grad()
def test_bidirectional_is_bidirectional():
    """Perturbing a late token MUST change earlier outputs (global visibility)."""
    layer = _make_layer(causal=False)
    info = _bidirectional_info()
    x = torch.randn(B, L, DIM, device=DEVICE)
    out_ref, _ = layer(x, info)

    p = L - 1  # perturb the very last token
    x2 = x.clone()
    x2[:, p, :] += torch.randn(B, DIM, device=DEVICE) * 3.0
    out_new, _ = layer(x2, info)

    diff_first = (out_ref[:, 0] - out_new[:, 0]).abs().max().item()
    assert diff_first > 1e-4, (
        f"bidirectional layer did not propagate last->first token: {diff_first}")
    print(f"[ok] bidirectional: perturbing last token changes first output "
          f"(diff={diff_first:.2e})")


def test_backward():
    """Autograd must flow to the input projection and the fast-weight params."""
    for causal in (False, True):
        layer = _make_layer(causal)
        x = torch.randn(B, L, DIM, device=DEVICE, requires_grad=True)
        info = {} if causal else _bidirectional_info()
        out, _ = layer(x, info)
        loss = out.float().pow(2).mean()
        loss.backward()
        assert x.grad is not None and torch.isfinite(x.grad).all(), causal
        assert layer.to_qkv.weight.grad is not None, causal
        assert layer.w0.grad is not None and torch.isfinite(layer.w0.grad).all(), causal
        print(f"[ok] backward (causal={causal}): finite grads on x / to_qkv / w0")


@torch.no_grad()
def test_state_chaining():
    """Fast weights returned from one call can be fed into the next via info."""
    layer = _make_layer(causal=False)
    info = _bidirectional_info()
    x = torch.randn(B, L, DIM, device=DEVICE)
    _, state = layer(x, info)
    # feed state back in
    info2 = dict(info)
    info2.update(state)
    out2, state2 = layer(x, info2)
    assert out2.shape == (B, L, DIM)
    assert torch.isfinite(out2).all()
    print("[ok] state chaining: returned {w0,w1,w2} accepted on next call")


@torch.no_grad()
def test_operator_level_causality():
    """Directly exercise the causal operator with chunk_size=1 controlled q,k,v."""
    torch.manual_seed(1)
    bh, l, d = 4, 16, 8
    dh = 8
    w0 = torch.randn(bh, d, dh, device=DEVICE) / d**0.5
    w1 = torch.randn(bh, dh, d, device=DEVICE) / dh**0.5
    w2 = torch.randn(bh, d, dh, device=DEVICE) / d**0.5
    q = torch.randn(bh, l, d, device=DEVICE)
    k = torch.randn(bh, l, d, device=DEVICE)
    v = torch.randn(bh, l, d, device=DEVICE)
    lr = torch.full((bh, l, 1), 0.01, device=DEVICE)

    out_ref, _, _, _ = causal_block_fast_weight_swish_glu(
        w0.clone(), w1.clone(), w2.clone(), q, k, v, lr, lr, lr,
        chunk_size=1, muon_update_steps=0)

    p = l // 2
    k2, v2 = k.clone(), v.clone()
    k2[:, p, :] += 5.0
    v2[:, p, :] += 5.0
    out_new, _, _, _ = causal_block_fast_weight_swish_glu(
        w0.clone(), w1.clone(), w2.clone(), q, k2, v2, lr, lr, lr,
        chunk_size=1, muon_update_steps=0)

    # apply-then-update: output i depends on k,v of chunks < i, so output<=p unchanged
    before = (out_ref[:, :p + 1] - out_new[:, :p + 1]).abs().max().item()
    after = (out_ref[:, p + 1:] - out_new[:, p + 1:]).abs().max().item()
    assert before < 1e-5, f"operator causality violated: {before}"
    assert after > 1e-4, f"operator perturbation had no effect: {after}"
    print(f"[ok] operator causal: diff(<=p)={before:.2e}  diff(>p)={after:.2e}")


if __name__ == "__main__":
    print(f"device = {DEVICE}, torch = {torch.__version__}")
    test_construct_and_shape()
    test_no_hard_triton_dependency()
    test_causal_is_causal()
    test_bidirectional_is_bidirectional()
    test_backward()
    test_state_chaining()
    test_operator_level_causality()
    print("\nAll TTT layer tests passed.")
