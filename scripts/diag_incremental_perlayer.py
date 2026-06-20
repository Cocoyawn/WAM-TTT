"""Pinpoint the incremental bug: teacher-forced, REAL checkpoint, per-layer.

Feed BOTH paths the SAME fixed token sequence (no argmax feedback) and compare,
layer by layer, position by position:
  forward(full_seq)          -- training/eval path, the ground truth
  incremental infer_step loop -- generate_incremental's per-token path

If per-layer |Δ| is round-off (~1e-2) the incremental APPLY is correct and the
eval breakage is pure argmax-feedback chaos (not fixable by a code bug, needs a
different integration). If |Δ| is large at some layer, infer_step has a real bug
there (e.g. attention KV-cache or TTT state), which IS fixable.
"""
import torch
from src.models.generator import ImageGeneratorTransformer

CKPT = ("/mnt/afs-h200/yuyangcheng/workplace/VLANeXt/VLANeXt_ablation_wm/"
        "ttt_chunk256_libero_mixed_clean/checkpoint_final.pt")
DEV = "cuda"
VOCAB, VLM_H, HID, DEPTH, HEADS = 131072, 2048, 768, 29, 12
CHUNK, MIX, N, T_CTX = 256, 4, 256, 16


def build():
    g = ImageGeneratorTransformer(
        vocab_size=VOCAB, vlm_hidden_size=VLM_H, hidden_size=HID, depth=DEPTH,
        num_heads=HEADS, max_seq_len=1024, mixer_type="ttt", mix_every_n=MIX,
        ttt_chunk_size=CHUNK, fallback_mixer="attention", ttt_use_cuda_kernel=False,
    ).to(DEV, torch.bfloat16).eval()
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)["model_state_dict"]
    gsd = {k[k.index("generator.") + 10:]: v for k, v in sd.items() if "generator." in k}
    g.load_state_dict(gsd, strict=False)
    return g


@torch.no_grad()
def incremental_teacher_forced(g, vlm, forced):
    """generate_incremental's exact per-block loop, but feed `forced` tokens
    (no argmax). forced: (B, N) the token AT each position (position t input)."""
    blocks = g.blocks
    relevant = vlm[-len(blocks):]
    dtype = next(g.parameters()).dtype
    states = [blk.infer_init(v.to(dtype=dtype)) for blk, v in zip(blocks, relevant)]
    B = vlm[0].shape[0]
    hs = [[] for _ in blocks]
    for t in range(forced.shape[1]):
        curr_id = forced[:, t:t + 1]
        x = g.token_emb(curr_id) + g.pos_embed[:, t:t + 1, :]
        for li, (blk, st) in enumerate(zip(blocks, states)):
            x = blk.infer_step(x, st)
            hs[li].append(x)
    return [torch.cat(h, dim=1) for h in hs]


if __name__ == "__main__":
    gs = torch.Generator(device=DEV).manual_seed(7)
    vlm = [torch.randn(1, T_CTX, VLM_H, generator=gs, device=DEV, dtype=torch.bfloat16)
           for _ in range(DEPTH)]
    B = 1
    g = build()
    # canonical token seq = full forward's own argmax free-run
    curr = torch.zeros((B, 1), dtype=torch.long, device=DEV)
    for _ in range(N):
        lg, _ = g(curr, vlm)
        curr = torch.cat([curr, torch.argmax(lg[:, -1:], dim=-1)], dim=1)
    forced = curr[:, :N]                          # [BOS, t1..t_{N-1}]
    # ground truth: one full forward over the fixed sequence
    _, hs_full = g(forced, vlm)
    # incremental teacher-forced over the SAME fixed sequence
    hs_inc = incremental_teacher_forced(g, vlm, forced)

    print(f"\nteacher-forced, real ckpt, seq len={forced.shape[1]}")
    print(f"{'layer':>5s} {'mixer':>10s} {'mean|Δ|':>12s} {'max|Δ|':>12s} {'ref_scale':>12s}")
    for li in range(DEPTH):
        a, b = hs_inc[li], hs_full[li]
        d = (a - b).abs()
        mt = g.blocks[li].mixer_type
        print(f"{li:5d} {mt:>10s} {float(d.mean()):12.4e} {float(d.max()):12.4e} "
              f"{float(b.abs().mean()):12.4e}")
