"""
B-level end-to-end confirmation: the incremental decode wired into the model
(ImageGeneratorTransformer.generate_incremental, the path predict_action now uses
when use_incremental_gen=True) reproduces the original O(n^2) AR-loop + line-887
full-forward, using the REAL TTT-mix checkpoint generator weights.

Per the chosen "generator-level fidelity": we load ONLY the generator weights from
checkpoint_final.pt (no Qwen, no sim), synthesize a correctly-shaped VLM context
(29 layers x (B, T_vlm, 2048)), and compare:
  1. token IDs            -- generate_incremental vs the AR argmax loop
  2. gen_hidden_states    -- the 29 per-layer tensors that actually feed the
                             diffusion action head (the load-bearing output)
against the bf16-vs-fp32 round-off floor (same weights, fp32) so any gap is shown
to be precision, not logic.

Deployed topology (from the ckpt): depth=29, hidden=768, heads=12, vocab=131072,
vlm_hidden=2048, chunk=256, mix_every_n=4 -> TTT at blocks 3/7/11/.../27, the rest
attention. So this exercises BOTH the TTT infer_step path AND the attention
KV-cache path of generate_incremental.

Run on a CLEAN GPU:
  TORCHDYNAMO_DISABLE=1 PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES=<clean> \
    /mnt/afs-h200/yuyangcheng/venvs/fla_triton32/bin/python -m scripts.equiv_predict_action_b
"""
import torch
from src.models.generator import ImageGeneratorTransformer

CKPT = ("/mnt/afs-h200/yuyangcheng/workplace/VLANeXt/VLANeXt_ablation_wm/"
        "ttt_chunk256_libero_mixed_clean/checkpoint_final.pt")
DEV = "cuda"
# from the checkpoint config + shapes
VOCAB, VLM_H, HID, DEPTH, HEADS = 131072, 2048, 768, 29, 12
CHUNK, MIX_EVERY = 256, 4
N_IMG_TOKENS = 256
T_CTX = 16          # synthetic VLM context length (predict_action passes the VLM
                    # hidden states; their seq length doesn't change the math here)


def build_generator(dtype, use_cuda_kernel):
    g = ImageGeneratorTransformer(
        vocab_size=VOCAB, vlm_hidden_size=VLM_H, hidden_size=HID, depth=DEPTH,
        num_heads=HEADS, max_seq_len=1024, mixer_type="ttt", mix_every_n=MIX_EVERY,
        ttt_chunk_size=CHUNK, fallback_mixer="attention",
        ttt_use_cuda_kernel=use_cuda_kernel,
    )
    return g.to(DEV, dtype).eval()


def load_generator_weights(gen):
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)["model_state_dict"]
    gsd = {}
    for k, v in sd.items():
        if "generator." in k:
            gk = k[k.index("generator.") + len("generator."):]
            gsd[gk] = v
    missing, unexpected = gen.load_state_dict(gsd, strict=False)
    miss = [m for m in missing if not m.endswith("_graph_cache")]
    assert not miss, f"missing generator keys: {miss[:10]}"
    assert not unexpected, f"unexpected keys: {unexpected[:10]}"
    return gen


@torch.no_grad()
def decode_full(gen, vlm_states, forced=None):
    """ORIGINAL predict_action path: O(n^2) AR loop, then line-887 full forward.
    forced=None -> argmax self-feed (deployment). forced=(B,256) -> teacher-forced
    (feed given tokens; isolates the operator from AR chaos)."""
    B = vlm_states[0].shape[0]
    curr = torch.zeros((B, 1), dtype=torch.long, device=DEV)
    for t in range(N_IMG_TOKENS):
        logits, _ = gen(curr, vlm_states)
        nxt = (forced[:, t:t + 1] if forced is not None
               else torch.argmax(logits[:, -1, :], dim=-1, keepdim=True))
        curr = torch.cat([curr, nxt], dim=1)
    gen_input = curr[:, :-1]
    _, gen_hidden = gen(gen_input, vlm_states)
    return curr[:, 1:], gen_hidden            # (B,256), list[(B,256,768)]


@torch.no_grad()
def decode_incremental_forced(gen, vlm_states, forced):
    """Teacher-forced incremental decode using the REAL block methods
    (infer_init / infer_step) -- same code path as generate_incremental, but we
    feed `forced` tokens instead of argmax so it lines up position-for-position
    with decode_full(forced). Returns the 29 per-layer hidden states."""
    blocks = gen.blocks
    relevant = vlm_states[-len(blocks):]
    dtype = next(gen.parameters()).dtype
    states = [blk.infer_init(v.to(dtype=dtype)) for blk, v in zip(blocks, relevant)]
    B = vlm_states[0].shape[0]
    curr_id = torch.zeros((B, 1), dtype=torch.long, device=DEV)
    hs = [[] for _ in blocks]
    for t in range(N_IMG_TOKENS):
        x = gen.token_emb(curr_id) + gen.pos_embed[:, t:t + 1, :]
        for li, (blk, st) in enumerate(zip(blocks, states)):
            x = blk.infer_step(x, st)
            hs[li].append(x)
        curr_id = forced[:, t:t + 1]
    return [torch.cat(h, dim=1) for h in hs]


def err(a, b):
    d = (a - b).abs()
    rel = d / (b.abs() + 1e-6)
    return d.max().item(), d.mean().item(), rel.mean().item()


if __name__ == "__main__":
    free, total = torch.cuda.mem_get_info()
    used_mb = (total - free) / 1e6
    print(f"== B: generate_incremental == full-recompute (predict_action path) | "
          f"REAL ckpt | depth{DEPTH} chunk{CHUNK} mix{MIX_EVERY} | GPU≈{used_mb:.0f}MiB"
          + ("  [WARN polluted]" if used_mb > 2000 else ""))

    g = torch.Generator(device=DEV).manual_seed(7)
    vlm_bf16 = [torch.randn(1, T_CTX, VLM_H, generator=g, device=DEV, dtype=torch.bfloat16)
                for _ in range(DEPTH)]

    gen = load_generator_weights(build_generator(torch.bfloat16, use_cuda_kernel=False))
    gen_cuda = load_generator_weights(build_generator(torch.bfloat16, use_cuda_kernel=True))
    gen32 = load_generator_weights(build_generator(torch.float32, use_cuda_kernel=False))
    vlm_fp32 = [v.float() for v in vlm_bf16]

    def seq_match(a, b):
        eq = (a == b)
        first = (~eq).float().argmax().item() if not eq.all() else -1
        return eq.float().mean().item() * 100, first

    def hs_err(A, Bl):
        ams = [err(a, b) for a, b in zip(A, Bl)]
        return (max(x[0] for x in ams),
                sum(x[1] for x in ams) / len(ams),
                sum(x[2] for x in ams) / len(ams))

    # ============================================================
    # TEST 1 (rigorous): TEACHER-FORCED gen_hidden_states equivalence.
    # Feed every path the SAME token sequence (fp32 self-fed) so the ONLY
    # difference is full-recompute generator.forward vs generate_incremental's
    # per-block infer_step. gen_hidden_states is what feeds the action head.
    # ============================================================
    forced, _ = decode_full(gen32, vlm_fp32)             # canonical token sequence
    _, hs_ff = decode_full(gen, vlm_bf16, forced=forced)
    _, hs_f32 = decode_full(gen32, vlm_fp32, forced=forced)
    hs_it = decode_incremental_forced(gen, vlm_bf16, forced)
    hs_ic = decode_incremental_forced(gen_cuda, vlm_bf16, forced)

    print("\n############ TEST 1: TEACHER-FORCED (isolates the operator) ############")
    print("-- gen_hidden_states error: the 29 tensors that feed the action head --")
    print(f"  {'pair':36s}   abs_max     abs_mean    rel_mean")
    for name, A, Bl in [
        ("incremental-torch vs full-bf16", hs_it, hs_ff),
        ("incremental-CUDA  vs full-bf16", hs_ic, hs_ff),
        ("incremental-CUDA  vs full-fp32", hs_ic, hs_f32),
        ("full-bf16         vs full-fp32", hs_ff, hs_f32),  # round-off floor
    ]:
        am, ame, rme = hs_err(A, Bl)
        print(f"  {name:36s}  {am:9.3e}  {ame:9.3e}  {rme:9.3e}")
    print("  ^ incremental-vs-bf16 <= bf16-vs-fp32 floor => generate_incremental is")
    print("    equivalent to the original predict_action generator path (precision, not logic).")

    # ============================================================
    # TEST 2 (deployment reality): FREE-RUNNING. argmax self-feed, exactly what
    # predict_action does. Shows token-sequence agreement under AR chaos (any
    # round-off snowballs; the SAME effect appears in bf16-vs-fp32, so divergence
    # here is precision, not a generate_incremental bug).
    # ============================================================
    ids_full, hs_full = decode_full(gen, vlm_bf16)
    ids_f32_fr, _ = decode_full(gen32, vlm_fp32)
    ids_it, _ = gen.generate_incremental(vlm_bf16, N_IMG_TOKENS)
    ids_ic, _ = gen_cuda.generate_incremental(vlm_bf16, N_IMG_TOKENS)

    print("\n############ TEST 2: FREE-RUNNING (predict_action's actual argmax loop) ############")
    print("-- 256-token sequence agreement (AR chaos; bf16-vs-fp32 is the floor) --")
    for name, ids, ref in [
        ("incremental-torch vs full-bf16", ids_it, ids_full),
        ("incremental-CUDA  vs full-bf16", ids_ic, ids_full),
        ("full-bf16         vs full-fp32", ids_full, ids_f32_fr),
    ]:
        pct, first = seq_match(ids, ref)
        tag = "IDENTICAL" if pct == 100.0 else f"1st div @tok{first}"
        print(f"  {name:34s}: {pct:6.2f}% match   {tag}")

