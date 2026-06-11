"""
Smoke test for the vendored GatedDeltaNet (src/models/fla) used as an ablation
baseline against the TTT layer.

Environment:
    Requires triton>=3.2 (the fla triton kernels need it). On the dev box this
    means running inside the isolated venv that shadows the system triton 3.0:

        bash src/models/run_fla_smoke.sh

    The script sets TORCHDYNAMO_DISABLE=1 because the custom torch 2.3 inductor
    backend is not compatible with triton 3.2 (the fla `@torch.compile` helpers
    fall back to eager, which is numerically identical).

What is checked:
  1. import of GatedDeltaNet (full triton-kernel dependency closure)
  2. forward (chunk mode, training)  -> shape + finite
  3. backward                        -> finite grads on input + projections
  4. inference path (fused_recurrent, short seq, no_grad)
  5. use_gate=False variant
"""

import torch

DEVICE = "cuda"


def _build(hidden_size=192, head_dim=64, num_heads=3, expand_v=2,
           use_gate=True, mode="chunk"):
    from src.models.fla.layers.gated_deltanet import GatedDeltaNet
    torch.manual_seed(0)
    return GatedDeltaNet(
        hidden_size=hidden_size, head_dim=head_dim, num_heads=num_heads,
        expand_v=expand_v, use_short_conv=True, use_gate=use_gate, mode=mode,
    ).to(DEVICE).to(torch.bfloat16)


def test_import():
    from src.models.fla.layers.gated_deltanet import GatedDeltaNet  # noqa: F401
    import triton
    print(f"[ok] import GatedDeltaNet (triton {triton.__version__})")


def test_forward_backward_chunk():
    m = _build()
    m.train()
    B, T, H = 2, 128, 192
    x = torch.randn(B, T, H, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
    o, _, _ = m(x)
    assert o.shape == (B, T, H), o.shape
    assert torch.isfinite(o).all(), "non-finite forward output"
    loss = o.float().pow(2).mean()
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all(), "bad input grad"
    assert torch.isfinite(m.q_proj.weight.grad).all(), "bad q_proj grad"
    print(f"[ok] chunk train fwd+bwd: out={tuple(o.shape)} finite grads")


@torch.no_grad()
def test_inference_fused_recurrent():
    # q_len <= 64 routes to fused_recurrent in eval mode.
    m = _build()
    m.eval()
    B, T, H = 2, 32, 192
    x = torch.randn(B, T, H, device=DEVICE, dtype=torch.bfloat16)
    o, _, _ = m(x)
    assert o.shape == (B, T, H), o.shape
    assert torch.isfinite(o).all(), "non-finite inference output"
    print(f"[ok] inference (fused_recurrent, T={T}): out={tuple(o.shape)} finite")


def test_no_gate_variant():
    m = _build(use_gate=False)
    m.train()
    B, T, H = 2, 128, 192
    x = torch.randn(B, T, H, device=DEVICE, dtype=torch.bfloat16)
    o, _, _ = m(x)
    assert o.shape == (B, T, H)
    assert torch.isfinite(o).all()
    print("[ok] use_gate=False variant: forward finite")


if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA required for fla triton kernels"
    print(f"device={DEVICE}, torch={torch.__version__}")
    test_import()
    test_forward_backward_chunk()
    test_inference_fused_recurrent()
    test_no_gate_variant()
    print("\nAll GatedDeltaNet smoke tests passed.")
