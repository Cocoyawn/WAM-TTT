"""Collect per-layer relative error for FREE-RUNNING decode (incremental vs
full-recompute), real mix checkpoint, to contrast with the teacher-forced curve.

teacher-forced (already have): same fixed tokens -> error is pure round-off,
stays <2.2% across 29 layers.
free-running (this script): each path argmax-self-feeds -> token sequences
diverge -> per-layer hidden error is amplified far beyond round-off.

Both paths produce (B, N, H) per-layer hidden; we compare position-wise:
  rel_err[layer] = mean|hs_incr - hs_full| / mean|hs_full|   (%)
Dumps docs/incremental_layer_error_freerun.json.
"""
import json, torch
from src.models.generator import ImageGeneratorTransformer

CKPT = ("/mnt/afs-h200/yuyangcheng/workplace/VLANeXt/VLANeXt_ablation_wm/"
        "ttt_chunk256_libero_mixed_clean/checkpoint_final.pt")
DEV = "cuda"
VOCAB, VLM_H, HID, DEPTH, HEADS = 131072, 2048, 768, 29, 12
CHUNK, MIX, N, T_CTX = 256, 4, 256, 16
TTT_LAYERS = [3, 7, 11, 15, 19, 23, 27]


def build():
    g = ImageGeneratorTransformer(
        vocab_size=VOCAB, vlm_hidden_size=VLM_H, hidden_size=HID, depth=DEPTH,
        num_heads=HEADS, max_seq_len=1024, mixer_type="ttt", mix_every_n=MIX,
        ttt_chunk_size=CHUNK, fallback_mixer="attention", ttt_use_cuda_kernel=False)
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)["model_state_dict"]
    g.load_state_dict({k[k.index("generator.") + 10:]: v for k, v in sd.items()
                       if "generator." in k}, strict=False)
    return g.to(DEV, torch.bfloat16).eval()


@torch.no_grad()
def full_freerun(g, vlm):
    curr = torch.zeros((1, 1), dtype=torch.long, device=DEV)
    for _ in range(N):
        lg, _ = g(curr, vlm)
        curr = torch.cat([curr, torch.argmax(lg[:, -1:], dim=-1)], dim=1)
    _, hs = g(curr[:, :-1], vlm)
    return curr[:, 1:N + 1], hs


if __name__ == "__main__":
    g = build()
    gs = torch.Generator(device=DEV).manual_seed(7)
    vlm = [torch.randn(1, T_CTX, VLM_H, generator=gs, device=DEV, dtype=torch.bfloat16)
           for _ in range(DEPTH)]

    ids_full, hs_full = full_freerun(g, vlm)
    ids_incr, hs_incr = g.generate_incremental(vlm, N)
    tok_match = float((ids_full == ids_incr).float().mean() * 100)

    rows = []
    for li in range(DEPTH):
        a, b = hs_incr[li], hs_full[li]
        rel = float((a - b).abs().mean() / (b.abs().mean() + 1e-9)) * 100
        rows.append({"layer": li, "mixer": "ttt" if li in TTT_LAYERS else "attention",
                     "rel_pct": rel})
        print(f"  L{li:2d} {'ttt' if li in TTT_LAYERS else 'attn':>4s}  rel={rel:7.2f}%")
    out = {"meta": {"mode": "free-running", "depth": DEPTH, "ttt_layers": TTT_LAYERS,
                    "token_match_pct": tok_match}, "layers": rows}
    json.dump(out, open("/mnt/afs-h200/yuyangcheng/workplace/VLANeXt/docs/"
                        "incremental_layer_error_freerun.json", "w"), indent=2)
    print(f"\ntoken match={tok_match:.1f}%  | wrote docs/incremental_layer_error_freerun.json")
