"""Verify the CUDA-Graph stale-buffer hypothesis for SR=0.

Hypothesis: infer_step's CUDA-graph cache is keyed by state['w0'].data_ptr().
Across episodes, infer_build_state allocates a NEW w0; PyTorch may REUSE the
freed address -> same data_ptr -> cache HIT on a graph captured with a DIFFERENT
episode's fast weights -> replay reads stale weights -> wrong conditioning ->
SR=0. The single-frame image compare never reuses across contexts, so it looked
perfect.

Test: build TWO different states (different ctx). Run infer_step on each via the
graph path; compare to the eager path (ground truth) for EACH. If episode-2's
graph output matches episode-1 instead of its own eager result, the bug is
confirmed.
"""
import torch
from src.models.generator import ImageGeneratorTransformer

CKPT = ("/mnt/afs-h200/yuyangcheng/workplace/VLANeXt/VLANeXt_ablation_wm/"
        "ttt_chunk256_libero_mixed_clean/checkpoint_final.pt")
DEV = "cuda"
VOCAB, VLM_H, HID, DEPTH, HEADS = 131072, 2048, 768, 29, 12
CHUNK, MIX, T_CTX = 256, 4, 16


def build():
    g = ImageGeneratorTransformer(
        vocab_size=VOCAB, vlm_hidden_size=VLM_H, hidden_size=HID, depth=DEPTH,
        num_heads=HEADS, max_seq_len=1024, mixer_type="ttt", mix_every_n=MIX,
        ttt_chunk_size=CHUNK, fallback_mixer="attention", ttt_use_cuda_kernel=True)
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)["model_state_dict"]
    g.load_state_dict({k[k.index("generator.") + 10:]: v for k, v in sd.items()
                       if "generator." in k}, strict=False)
    return g.to(DEV, torch.bfloat16).eval()


@torch.no_grad()
def first_ttt_block(g):
    for blk in g.blocks:
        if getattr(blk, "mixer_type", None) == "ttt":
            return blk
    return None


if __name__ == "__main__":
    g = build()
    blk = first_ttt_block(g)
    gen = torch.Generator(device=DEV).manual_seed(1)

    # two DIFFERENT contexts -> two different fast-weight states (VLM dim 2048,
    # blk.vlm_proj maps 2048 -> 768 then infer_build_state expects mixer-dim ctx)
    ctxA = torch.randn(1, T_CTX, VLM_H, generator=gen, device=DEV, dtype=torch.bfloat16)
    ctxB = torch.randn(1, T_CTX, VLM_H, generator=gen, device=DEV, dtype=torch.bfloat16) * 3.0
    x = torch.randn(1, 1, HID, generator=gen, device=DEV, dtype=torch.bfloat16)

    # --- episode A ---
    stA = blk.attn.infer_build_state(blk.vlm_proj(ctxA))
    # eager ground truth for A
    blk.attn._use_infer_graph = False
    eagerA = blk.attn.infer_step(blk.norm1(x) if False else x, stA).clone()
    # graph path for A (captures graph keyed by stA['w0'].data_ptr())
    blk.attn._use_infer_graph = True
    blk.attn._graph_cache = {}
    graphA = blk.attn.infer_step(x, stA).clone()
    ptrA = stA["w0"].data_ptr()

    # --- episode B (free A's state first to encourage address reuse) ---
    del stA
    torch.cuda.empty_cache()
    stB = blk.attn.infer_build_state(blk.vlm_proj(ctxB))
    ptrB = stB["w0"].data_ptr()
    blk.attn._use_infer_graph = False
    eagerB = blk.attn.infer_step(x, stB).clone()       # B's true answer
    blk.attn._use_infer_graph = True
    graphB = blk.attn.infer_step(x, stB).clone()        # B via graph path

    def rel(a, b):
        return float((a - b).abs().mean() / (b.abs().mean() + 1e-9))

    print(f"\nw0 data_ptr A={ptrA}  B={ptrB}  SAME={ptrA == ptrB}")
    print(f"episode A: graph vs eager           rel={rel(graphA, eagerA):.4e}  (should be ~round-off)")
    print(f"episode B: graph vs eager(B's own)  rel={rel(graphB, eagerB):.4e}  (BUG if large)")
    print(f"episode B: graph vs eager(A's)      rel={rel(graphB, eagerA):.4e}  (BUG-confirm if ~0)")
    print(f"sanity   : eagerA vs eagerB         rel={rel(eagerA, eagerB):.4e}  (should be large: diff ctx)")
