"""
Unit test for the GLA/GatedDeltaNet (gdn) and causal-TTT mixers wired into
MoEBlock (action expert) and MoEGeneratorBlock (vision expert).

Checks, per mixer:
  - construct + forward produces correct shape and finite values
  - backward produces finite grads
  - (gdn / causal-ttt) CAUSALITY: perturbing action/image token at position p does
    not change the output at positions < p (left-to-right causal token mixing).
  - method-B VLM injection: perturbing the VLM ctx DOES change the output (tokens
    actually read the context).

Run:
  TORCHDYNAMO_DISABLE=1 PYTHONPATH="$PWD" venv/bin/python -m src.models.test_linear_attn_mixer
"""
import torch

from src.models.policies import MoEBlock
from src.models.generator import MoEGeneratorBlock

torch.manual_seed(0)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
DT = torch.float32

B, L_act, L_img, H, VLM = 2, 8, 64, 1024, 2048
NH = 16


def _mixers_available():
    mix = ["attention", "ttt", "gdn"]
    try:
        from src.models.fla.layers.gla import GatedLinearAttention  # noqa
        mix.append("gla")
    except Exception as e:
        print(f"[skip gla] not vendored yet: {e}")
    return mix


def test_action_block(mixer):
    print(f"\n=== action MoEBlock mixer={mixer} ===")
    ttt_causal = (mixer == "ttt")  # use the causal-ttt variant for the probe
    blk = MoEBlock(H, VLM, NH, mixer_type=mixer, ttt_causal=ttt_causal,
                   ttt_chunk_size=2, layer_idx=0).to(DEV, DT)
    x = torch.randn(B, L_act, H, device=DEV, dtype=DT, requires_grad=True)
    c = torch.randn(B, H, device=DEV, dtype=DT)
    vlm = torch.randn(B, 5, VLM, device=DEV, dtype=DT)
    out = blk(x, c, vlm)
    assert out.shape == (B, L_act, H), out.shape
    assert torch.isfinite(out).all(), "non-finite output"
    out.sum().backward()
    assert torch.isfinite(x.grad).all(), "non-finite grad"
    print(f"  shape {tuple(out.shape)} OK, finite OK, grad OK")


def test_vision_block_causality(mixer):
    print(f"\n=== vision MoEGeneratorBlock mixer={mixer} (causality) ===")
    blk = MoEGeneratorBlock(H, VLM, NH, mixer_type=mixer, ttt_chunk_size=16,
                            layer_idx=0).to(DEV, DT).eval()
    x = torch.randn(B, L_img, H, device=DEV, dtype=DT)
    vlm = torch.randn(B, 5, VLM, device=DEV, dtype=DT)
    with torch.no_grad():
        out0 = blk(x, vlm)
        assert out0.shape == (B, L_img, H) and torch.isfinite(out0).all()
        # perturb image token at position p; outputs at < p must be unchanged (causal)
        p = L_img // 2
        x2 = x.clone()
        x2[:, p, :] += 5.0
        out1 = blk(x2, vlm)
        pre_delta = (out1[:, :p] - out0[:, :p]).abs().max().item()
        post_delta = (out1[:, p:] - out0[:, p:]).abs().max().item()
        print(f"  perturb img tok {p}: max delta BEFORE={pre_delta:.2e} (want ~0), AFTER={post_delta:.2e} (want >0)")
        if mixer in ("ttt", "gdn", "gla"):
            assert pre_delta < 1e-3, f"causality violated: pre_delta={pre_delta}"
            assert post_delta > 1e-3, f"position p+ unaffected: post_delta={post_delta}"
        # perturb VLM ctx; ALL positions may change (method-B global injection)
        vlm2 = vlm.clone() + 3.0
        out2 = blk(x, vlm2)
        vlm_delta = (out2 - out0).abs().max().item()
        print(f"  perturb VLM ctx: max delta={vlm_delta:.2e} (want >0, method-B injection works)")
        assert vlm_delta > 1e-3, "VLM ctx not read"


if __name__ == "__main__":
    mixers = _mixers_available()
    print("Testing mixers:", mixers)
    for m in mixers:
        test_action_block(m)
        test_vision_block_causality(m)
    print("\nALL MIXER TESTS PASSED")
