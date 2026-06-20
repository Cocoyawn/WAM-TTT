"""Diagnose why CUDA+incremental gives all-zero SR in real eval.

Real eval is FREE-RUNNING (argmax self-feed). We compare, with the REAL mix
checkpoint generator, the gen_hidden_states that feed the action head, produced by:
  (1) full-recompute free-run  (what torch eval does, the known-good baseline)
  (2) generate_incremental      (use_incremental_gen=True, torch infer_step)
  (3) generate_incremental CUDA (use_incremental_gen=True, ttt_use_cuda_kernel=True)
Isolates whether the breakage is the incremental logic or the CUDA kernel, and by
how much the action-head input diverges. No sim, no Qwen.
"""
import torch
from src.models.generator import ImageGeneratorTransformer

CKPT = ("/mnt/afs-h200/yuyangcheng/workplace/VLANeXt/VLANeXt_ablation_wm/"
        "ttt_chunk256_libero_mixed_clean/checkpoint_final.pt")
DEV = "cuda"
VOCAB, VLM_H, HID, DEPTH, HEADS = 131072, 2048, 768, 29, 12
CHUNK, MIX, N = 256, 4, 256
T_CTX = 16


def build(dtype, cuda):
    g = ImageGeneratorTransformer(
        vocab_size=VOCAB, vlm_hidden_size=VLM_H, hidden_size=HID, depth=DEPTH,
        num_heads=HEADS, max_seq_len=1024, mixer_type="ttt", mix_every_n=MIX,
        ttt_chunk_size=CHUNK, fallback_mixer="attention", ttt_use_cuda_kernel=cuda,
    ).to(DEV, dtype).eval()
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)["model_state_dict"]
    gsd = {k[k.index("generator.") + 10:]: v for k, v in sd.items() if "generator." in k}
    g.load_state_dict(gsd, strict=False)
    return g


@torch.no_grad()
def full_freerun(g, vlm):
    """torch eval's path: AR argmax loop, then line-887 full forward."""
    B = vlm[0].shape[0]
    curr = torch.zeros((B, 1), dtype=torch.long, device=DEV)
    for _ in range(N):
        logits, _ = g(curr, vlm)
        nxt = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        curr = torch.cat([curr, nxt], dim=1)
    _, hs = g(curr[:, :-1], vlm)
    return curr[:, 1:], hs


def stats(name, a, ref):
    bad = sum(int(torch.isnan(x).any() or torch.isinf(x).any()) for x in a)
    d = [float((x - y).abs().mean()) for x, y in zip(a, ref)]
    print(f"  {name:28s} layers={len(a)} nan/inf_layers={bad} "
          f"mean|Δ|={sum(d)/len(d):.4e} max_layer|Δ|={max(d):.4e}")


if __name__ == "__main__":
    gseed = torch.Generator(device=DEV).manual_seed(7)
    vlm = [torch.randn(1, T_CTX, VLM_H, generator=gseed, device=DEV, dtype=torch.bfloat16)
           for _ in range(DEPTH)]

    g = build(torch.bfloat16, cuda=False)
    g_cuda = build(torch.bfloat16, cuda=True)

    ids_full, hs_full = full_freerun(g, vlm)
    res_t = g.generate_incremental(vlm, N)
    res_c = g_cuda.generate_incremental(vlm, N)

    print(f"\nfull-recompute tokens[:12]   = {ids_full[0,:12].tolist()}")
    if res_t is None:
        print("  !! generate_incremental (torch) returned None")
    else:
        ids_t, hs_t = res_t
        print(f"incremental-torch tokens[:12]= {ids_t[0,:12].tolist()}")
        same = int((ids_t == ids_full).float().mean() * 100)
        print(f"  token seq match vs full = {same}%")
        stats("incremental-torch hs", hs_t, hs_full)
    if res_c is None:
        print("  !! generate_incremental (CUDA) returned None")
    else:
        ids_c, hs_c = res_c
        print(f"incremental-CUDA  tokens[:12]= {ids_c[0,:12].tolist()}")
        stats("incremental-CUDA hs", hs_c, hs_full)
    print(f"\nfull-recompute hs: layers={len(hs_full)} "
          f"nan/inf={sum(int(torch.isnan(x).any()) for x in hs_full)} "
          f"sample mean={float(hs_full[-1].abs().mean()):.4e}")
