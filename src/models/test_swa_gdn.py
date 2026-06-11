"""Tests for the SWA (sliding-window attention) vision mixer + [SWA,SWA,SWA,GDN] stack.

Run:  TORCHDYNAMO_DISABLE=1 python -m pytest src/models/test_swa_gdn.py -q
or:   TORCHDYNAMO_DISABLE=1 python src/models/test_swa_gdn.py
"""
import torch

from src.models.generator import (
    build_swa_causal_mask, _mixer_at, MoEGeneratorBlock, ImageGeneratorTransformer,
)


def test_swa_mask_semantics():
    """Window=3, 6 image tokens, 2 ctx tokens. Verify exactly which keys are visible."""
    T_img, T_ctx, W = 6, 2, 3
    m = build_swa_causal_mask(T_img, T_ctx, W, device="cpu", dtype=torch.float32)
    assert m.shape == (T_img, T_img + T_ctx)
    allowed = (m == 0.0)
    # ctx columns (last T_ctx) always visible for every query
    assert allowed[:, T_img:].all(), "VLM ctx must be globally visible"
    # image->image: query i sees keys [i-W+1 .. i]
    for i in range(T_img):
        for j in range(T_img):
            expect = (j <= i) and (j >= i - (W - 1))
            assert bool(allowed[i, j]) == expect, f"q{i} k{j}: expected {expect}"
    # spot-checks: no future, no beyond-window
    assert m[0, 1] == float("-inf")          # q0 can't see future k1
    assert m[5, 1] == float("-inf")          # q5 (window 3..5) can't see old k1
    assert m[5, 3] == 0.0 and m[5, 5] == 0.0  # q5 sees k3,k4,k5
    print("[ok] swa mask semantics")


def test_swa_window_full_equals_causal():
    """window_size >= T_img must degrade to plain causal (our baseline)."""
    T_img, T_ctx = 8, 3
    swa = build_swa_causal_mask(T_img, T_ctx, window_size=999, device="cpu", dtype=torch.float32)
    # build the plain causal mask the same way generator.forward does for 'attention'
    causal = torch.zeros((T_img, T_img + T_ctx))
    fut = torch.triu(torch.ones((T_img, T_img), dtype=torch.bool), diagonal=1)
    causal[:, :T_img].masked_fill_(fut, float("-inf"))
    assert torch.equal(swa, causal), "wide window must equal plain causal"
    print("[ok] wide window == causal")


def test_mixer_at_swa_fallback():
    """[SWA,SWA,SWA,GDN] interleave: layers 0,1,2 -> swa; layer 3 -> gdn (mix_every_n=4)."""
    got = [_mixer_at(i, "gdn", 4, fallback_mixer="swa") for i in range(8)]
    assert got == ["swa", "swa", "swa", "gdn", "swa", "swa", "swa", "gdn"], got
    # default fallback stays 'attention' (classic [A,A,A,X])
    got2 = [_mixer_at(i, "gdn", 4) for i in range(4)]
    assert got2 == ["attention", "attention", "attention", "gdn"], got2
    print("[ok] _mixer_at swa fallback")


def test_swa_block_forward_grad():
    """SWA MoEGeneratorBlock: shape / finite / grad."""
    torch.manual_seed(0)
    B, T_img, H, V = 2, 16, 64, 32
    blk = MoEGeneratorBlock(H, vlm_hidden_size=H, num_heads=4, mixer_type="swa", swa_window_size=4)
    x = torch.randn(B, T_img, H, requires_grad=True)
    vlm = torch.randn(B, 5, H)
    out = blk(x, vlm)
    assert out.shape == (B, T_img, H)
    assert torch.isfinite(out).all()
    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    print("[ok] swa block forward+grad")


def test_swa_block_causality():
    """A token's output must not depend on FUTURE tokens (causal), and must not depend
    on tokens OUTSIDE its window."""
    torch.manual_seed(0)
    B, T_img, H = 1, 12, 64
    W = 3
    blk = MoEGeneratorBlock(H, vlm_hidden_size=H, num_heads=4, mixer_type="swa", swa_window_size=W).eval()
    x = torch.randn(B, T_img, H)
    vlm = torch.randn(B, 4, H)
    with torch.no_grad():
        base = blk(x, vlm)
        # perturb a FUTURE token (pos 8); output at pos 4 must be unchanged
        x2 = x.clone(); x2[:, 8] += 5.0
        out2 = blk(x2, vlm)
        assert torch.allclose(base[:, 4], out2[:, 4], atol=1e-5), "pos4 changed by future pos8 (not causal!)"
        # perturb an OLD out-of-window token (pos 0); output at pos 8 (window 6..8) must be unchanged
        x3 = x.clone(); x3[:, 0] += 5.0
        out3 = blk(x3, vlm)
        assert torch.allclose(base[:, 8], out3[:, 8], atol=1e-5), "pos8 changed by out-of-window pos0"
    print("[ok] swa causality + window isolation")


def test_swa_gdn_full_generator():
    """End-to-end ImageGeneratorTransformer with [SWA,SWA,SWA,GDN] stack: real
    forward + backward, finite grads. Requires GPU (GatedDeltaNet triton kernels
    don't run on CPU); skips with a notice if no CUDA."""
    if not torch.cuda.is_available():
        print("[skip] swa+gdn e2e needs GPU (GDN triton kernel); CPU-only env")
        return
    dev = "cuda"
    torch.manual_seed(0)
    V, H, depth = 64, 64, 8
    gen = ImageGeneratorTransformer(
        vocab_size=V, vlm_hidden_size=H, hidden_size=H, depth=depth, num_heads=4,
        max_seq_len=64, mixer_type="gdn", mix_every_n=4,
        fallback_mixer="swa", swa_window_size=8,
    ).to(dev)
    kinds = [b.mixer_type for b in gen.blocks]
    assert kinds == ["swa","swa","swa","gdn","swa","swa","swa","gdn"], kinds
    input_ids = torch.randint(0, V, (2, 16), device=dev)
    vlm_states = [torch.randn(2, 5, H, device=dev) for _ in range(depth)]
    logits, _ = gen(input_ids, vlm_states)   # forward returns (logits, hidden_states)
    assert torch.isfinite(logits).all(), "non-finite logits"
    logits.sum().backward()
    g = gen.blocks[3].attn  # the gdn block
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in g.parameters())
    print("[ok] swa+gdn generator e2e (GPU) forward+backward, stack =", kinds)


if __name__ == "__main__":
    test_swa_mask_semantics()
    test_swa_window_full_equals_causal()
    test_mixer_at_swa_fallback()
    test_swa_block_forward_grad()
    test_swa_block_causality()
    test_swa_gdn_full_generator()
    print("\nALL SWA+GDN TESTS PASSED")
