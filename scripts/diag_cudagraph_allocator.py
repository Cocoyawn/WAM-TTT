"""Verify the CUDA-Graph stale-address crash is driven by the caching allocator
reorganizing memory. Force graph ON, repeatedly run incremental decode with a
FRESH fast-weight state each iter (like eval's per-step predict_action) plus
deliberate memory churn, and print allocator counters each iter. We expect the
crash (illegal memory access / CUBLAS) to coincide with an allocator event:
a jump in num_alloc_retries, or reserved/segment changing (free+realloc), or an
empty_cache. No crash for the first iters (addresses still valid), then boom.
"""
import torch, gc, traceback
from src.models.generator import ImageGeneratorTransformer

CKPT = ("/mnt/afs-h200/yuyangcheng/workplace/VLANeXt/VLANeXt_ablation_wm/"
        "ttt_chunk256_libero_mixed_clean/checkpoint_final.pt")
DEV = "cuda"
VOCAB, VLM_H, HID, DEPTH, HEADS = 131072, 2048, 768, 29, 12
CHUNK, MIX, T_CTX = 256, 4, 16
N_IMG = 256
ITERS = 60


def build():
    g = ImageGeneratorTransformer(
        vocab_size=VOCAB, vlm_hidden_size=VLM_H, hidden_size=HID, depth=DEPTH,
        num_heads=HEADS, max_seq_len=1024, mixer_type="ttt", mix_every_n=MIX,
        ttt_chunk_size=CHUNK, fallback_mixer="attention", ttt_use_cuda_kernel=True)
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)["model_state_dict"]
    g.load_state_dict({k[k.index("generator.") + 10:]: v for k, v in sd.items()
                       if "generator." in k}, strict=False)
    return g.to(DEV, torch.bfloat16).eval()


def stats():
    s = torch.cuda.memory_stats()
    return dict(
        retries=s.get("num_alloc_retries", 0),
        ooms=s.get("num_ooms", 0),
        segs=s.get("segment.all.current", 0),
        reserved_mb=s.get("reserved_bytes.all.current", 0) / 1e6,
        alloc_mb=s.get("allocated_bytes.all.current", 0) / 1e6,
    )


@torch.no_grad()
def main():
    g = build()
    # FORCE graph ON for every TTT block (override the new default-off fix)
    for blk in g.blocks:
        if getattr(blk, "mixer_type", None) == "ttt":
            blk.attn._use_infer_graph = True
            blk.attn._graph_cache = {}

    gen = torch.Generator(device=DEV).manual_seed(0)
    prev = stats()
    print(f"{'iter':>4} {'retries':>8} {'segs':>6} {'reserved_MB':>12} {'alloc_MB':>10} {'event':>20}")
    for it in range(ITERS):
        # fresh context each iter -> fresh fast weights (new w0/w1/w2), mimicking
        # eval's per-step predict_action where infer_build_state runs anew.
        vlm = [torch.randn(1, T_CTX, VLM_H, generator=gen, device=DEV, dtype=torch.bfloat16)
               for _ in range(DEPTH)]
        # memory churn: alloc + free odd-sized scratch to fragment the pool,
        # like eval's VLM/diffusion/sim intermediates between predict_action calls.
        scratch = [torch.empty(int(3.1e6) + it * 5000, device=DEV, dtype=torch.bfloat16)
                   for _ in range(8)]
        del scratch
        try:
            ids, _ = g.generate_incremental(vlm, N_IMG)
            torch.cuda.synchronize()
            ok = True
        except Exception as e:
            cur = stats()
            print(f"{it:>4} {cur['retries']:>8} {cur['segs']:>6} {cur['reserved_mb']:>12.1f} "
                  f"{cur['alloc_mb']:>10.1f} {'*** CRASH ***':>20}")
            print(f"\n--- exception at iter {it} ---")
            print(type(e).__name__, str(e)[:160])
            print(f"allocator delta vs prev: retries {prev['retries']}->{cur['retries']}  "
                  f"segs {prev['segs']}->{cur['segs']}  reserved {prev['reserved_mb']:.0f}->{cur['reserved_mb']:.0f}MB")
            return
        cur = stats()
        ev = []
        if cur["retries"] > prev["retries"]:
            ev.append(f"retry+{cur['retries']-prev['retries']}")
        if cur["segs"] != prev["segs"]:
            ev.append(f"seg{cur['segs']-prev['segs']:+d}")
        if abs(cur["reserved_mb"] - prev["reserved_mb"]) > 50:
            ev.append(f"resv{cur['reserved_mb']-prev['reserved_mb']:+.0f}")
        evs = ",".join(ev) if ev else "-"
        if it % 5 == 0 or ev:
            print(f"{it:>4} {cur['retries']:>8} {cur['segs']:>6} {cur['reserved_mb']:>12.1f} "
                  f"{cur['alloc_mb']:>10.1f} {evs:>20}")
        prev = cur
    print(f"\nNo crash in {ITERS} iters (graph forced ON). "
          f"final retries={prev['retries']} segs={prev['segs']} reserved={prev['reserved_mb']:.0f}MB")


if __name__ == "__main__":
    main()
