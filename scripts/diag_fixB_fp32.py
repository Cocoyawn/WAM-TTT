"""Validate the premise of Fix B before coding it.

Fix B hypothesis: running the incremental decode in fp32 makes it reproduce the
full-recompute token sequence (argmax stops flipping), restoring correct
gen_hidden_states. This ONLY holds if compared against the SAME-precision
full-recompute. So we test, FREE-RUNNING (real eval mode), the token-seq match:

  (1) fp32 full-recompute   (the fp32 ground truth)
  (2) fp32 incremental      (Fix B)            -> match vs (1) ?
  (3) bf16 full-recompute   (what torch eval actually uses)
  (4) fp32 incremental vs bf16 full-recompute  -> does Fix B match the eval baseline?

If (2)~100% and (4) high, Fix B works. If (2) high but (4) low, fp32 incremental
reproduces fp32-full but the eval baseline is bf16-full -> need bf16-full as the
reference (i.e. run the *whole* generator fp32 in eval too).
"""
import torch
from src.models.generator import ImageGeneratorTransformer

CKPT = ("/mnt/afs-h200/yuyangcheng/workplace/VLANeXt/VLANeXt_ablation_wm/"
        "ttt_chunk256_libero_mixed_clean/checkpoint_final.pt")
DEV = "cuda"
VOCAB, VLM_H, HID, DEPTH, HEADS = 131072, 2048, 768, 29, 12
CHUNK, MIX, N, T_CTX = 256, 4, 256, 16


def build(dtype):
    g = ImageGeneratorTransformer(
        vocab_size=VOCAB, vlm_hidden_size=VLM_H, hidden_size=HID, depth=DEPTH,
        num_heads=HEADS, max_seq_len=1024, mixer_type="ttt", mix_every_n=MIX,
        ttt_chunk_size=CHUNK, fallback_mixer="attention", ttt_use_cuda_kernel=False,
    ).to(DEV, dtype).eval()
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)["model_state_dict"]
    gsd = {k[k.index("generator.") + 10:]: v for k, v in sd.items() if "generator." in k}
    g.load_state_dict(gsd, strict=False)
    return g


@torch.no_grad()
def full_freerun(g, vlm):
    B = vlm[0].shape[0]
    curr = torch.zeros((B, 1), dtype=torch.long, device=DEV)
    for _ in range(N):
        lg, _ = g(curr, vlm)
        curr = torch.cat([curr, torch.argmax(lg[:, -1:], dim=-1)], dim=1)
    return curr[:, 1:]


@torch.no_grad()
def incr_freerun(g, vlm):
    ids, _ = g.generate_incremental(vlm, N)
    return ids


def match(a, b):
    return float((a == b).float().mean() * 100)


if __name__ == "__main__":
    gs = torch.Generator(device=DEV).manual_seed(7)
    vlm_bf16 = [torch.randn(1, T_CTX, VLM_H, generator=gs, device=DEV, dtype=torch.bfloat16)
                for _ in range(DEPTH)]
    vlm_fp32 = [v.float() for v in vlm_bf16]

    g32 = build(torch.float32)
    g16 = build(torch.bfloat16)

    ids_full32 = full_freerun(g32, vlm_fp32)
    ids_incr32 = incr_freerun(g32, vlm_fp32)
    ids_full16 = full_freerun(g16, vlm_bf16)
    ids_incr16 = incr_freerun(g16, vlm_bf16)

    print("\nfree-running token-seq match (%):")
    print(f"  (2) fp32 incr  vs  (1) fp32 full   = {match(ids_incr32, ids_full32):6.1f}%   <- Fix B core")
    print(f"  (4) fp32 incr  vs  (3) bf16 full   = {match(ids_incr32, ids_full16):6.1f}%   <- vs eval baseline")
    print(f"      bf16 incr  vs      bf16 full   = {match(ids_incr16, ids_full16):6.1f}%   <- current (broken)")
    print(f"      fp32 full  vs      bf16 full   = {match(ids_full32, ids_full16):6.1f}%   <- precision-only chaos")
    print(f"\nfirst-divergence index (fp32 incr vs fp32 full):")
    eq = (ids_incr32 == ids_full32)
    fd = int((~eq).float().argmax()) if not eq.all() else -1
    print(f"  {fd}  (-1 == identical)")
