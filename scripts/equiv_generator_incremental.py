"""
A-level end-to-end equivalence: incremental TTT decode  ==  full-recompute decode.

We do NOT touch predict_action. Instead we build a REAL ImageGeneratorTransformer
(mixer_type='ttt', chunk_size=256 -- the deployed config) and run the *exact*
predict_action autoregressive loop two ways:

  reference  : generator.forward(curr_ids, ctx) on the growing prefix, argmax-feed
               (this is byte-for-byte what VLANeXt.predict_action does today).
  incremental: per-layer infer_build_state(ctx) ONCE, then infer_step on the single
               new token each step, with the surrounding block math (norm1, residual,
               norm2+mlp, final norm+head) replicated. argmax-feed.

If positions are truly independent at chunk>=seq (CLAIM1), the incremental decode
must reproduce the full-recompute decode: identical 256-token argmax sequence and
logits within bf16 round-off. We test both the torch and CUDA infer_step paths,
and use an fp32 full-recompute as the high-precision ground truth.

Run on a CLEAN GPU:
  TORCHDYNAMO_DISABLE=1 PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES=<clean> \
    /mnt/afs-h200/yuyangcheng/venvs/fla_triton32/bin/python -m scripts.equiv_generator_incremental
"""
import torch
import torch.nn.functional as F
from einops import rearrange
from src.models.generator import ImageGeneratorTransformer

DEV = "cuda"
VOCAB, VLM_H, HID, DEPTH, HEADS = 1024, 768, 768, 12, 12
N_IMG_TOKENS = 256          # predict_action's num_img_tokens
CHUNK = 256                 # deployed TTT chunk -> single chunk -> positions independent
T_CTX = 16                  # VLM context tokens


def build_generator(dtype):
    torch.manual_seed(0)
    g = ImageGeneratorTransformer(
        vocab_size=VOCAB, vlm_hidden_size=VLM_H, hidden_size=HID, depth=DEPTH,
        num_heads=HEADS, max_seq_len=1024, mixer_type="ttt", mix_every_n=1,
        ttt_chunk_size=CHUNK, ttt_use_cuda_kernel=False,
    ).to(DEV, dtype).eval()
    return g


@torch.no_grad()
def decode_full(gen, vlm_states, forced=None):
    """predict_action's loop: forward on the growing prefix.
    forced=None -> free-running (argmax-feed, what predict_action does).
    forced=(B,256) longs -> teacher-forced: feed the given tokens; isolates the
    operator from AR chaos by giving every path the SAME prefix."""
    B = vlm_states[0].shape[0]
    curr = torch.zeros((B, 1), dtype=torch.long, device=DEV)
    last_logits = []
    out_ids = []
    for t in range(N_IMG_TOKENS):
        logits, _ = gen(curr, vlm_states)
        last_logits.append(logits[:, -1, :].float())
        nxt = (forced[:, t:t + 1] if forced is not None
               else torch.argmax(logits[:, -1, :], dim=-1, keepdim=True))
        out_ids.append(nxt)
        curr = torch.cat([curr, nxt], dim=1)
    return torch.cat(out_ids, dim=1), torch.stack(last_logits, dim=1)


@torch.no_grad()
def decode_incremental(gen, vlm_states, use_cuda, forced=None):
    """Same loop, decoded incrementally: per-layer state built once (VLM
    pre-update), infer_step on the single new token, block residual/mlp
    replicated around the mixer. O(1) TTT work per step. forced as in decode_full."""
    B = vlm_states[0].shape[0]
    blocks = gen.blocks
    relevant = vlm_states[-len(blocks):]
    # build per-layer fast-weight state ONCE (ctx already projected by vlm_proj)
    states = []
    for blk, vfeat in zip(blocks, relevant):
        blk.attn.use_cuda_kernel = use_cuda          # toggle infer_step path
        ctx = blk.vlm_proj(vfeat.to(next(gen.parameters()).dtype))
        states.append(blk.attn.infer_build_state(ctx))

    curr_id = torch.zeros((B, 1), dtype=torch.long, device=DEV)  # initial token
    out_ids, last_logits = [], []
    for t in range(N_IMG_TOKENS):
        # token_emb + absolute pos t  (position-dependent: must index by t)
        x = gen.token_emb(curr_id) + gen.pos_embed[:, t:t + 1, :]
        for blk, st in zip(blocks, states):
            x_norm = blk.norm1(x)
            attn_out = blk.attn.infer_step(x_norm, st)
            x = x + attn_out
            x = x + blk.mlp(blk.norm2(x))
        logits = gen.head(gen.norm_final(x))         # (B,1,V)
        last_logits.append(logits[:, -1, :].float())
        nxt = (forced[:, t:t + 1] if forced is not None
               else torch.argmax(logits[:, -1, :], dim=-1, keepdim=True))
        out_ids.append(nxt)
        curr_id = nxt                                # feed the new token next step
    return torch.cat(out_ids, dim=1), torch.stack(last_logits, dim=1)


def err(a, b):
    d = (a - b).abs()
    rel = d / (b.abs() + 1e-6)
    return d.max().item(), d.mean().item(), rel.max().item(), rel.mean().item()


if __name__ == "__main__":
    free, total = torch.cuda.mem_get_info()
    used_mb = (total - free) / 1e6
    print(f"== A: incremental-decode == full-recompute-decode | ttt chunk{CHUNK} | "
          f"depth{DEPTH} | {N_IMG_TOKENS}-token AR | GPU used≈{used_mb:.0f}MiB"
          + ("  [WARN polluted]" if used_mb > 2000 else ""))

    g = torch.Generator(device=DEV).manual_seed(7)
    vlm_bf16 = [torch.randn(1, T_CTX, VLM_H, generator=g, device=DEV, dtype=torch.bfloat16)
                for _ in range(DEPTH)]

    gen = build_generator(torch.bfloat16)
    gen32 = build_generator(torch.float32)
    gen32.load_state_dict(gen.state_dict())  # identical weights, fp32 reference
    vlm_fp32 = [v.float() for v in vlm_bf16]

    def seq_match(a, b):
        eq = (a == b)
        first = (~eq).float().argmax().item() if not eq.all() else -1
        return eq.float().mean().item() * 100, first

    # ============================================================
    # TEST 1 (rigorous): TEACHER-FORCED operator equivalence.
    # Feed EVERY path the SAME token prefix (the fp32 free-run sequence) so the
    # ONLY difference is full-recompute-mixer vs incremental infer_step. This
    # isolates the operator from AR chaos (argmax amplifies any round-off).
    # ============================================================
    forced, _ = decode_full(gen32, vlm_fp32)         # canonical token sequence
    ff_ids, ff_lg = decode_full(gen, vlm_bf16, forced=forced)
    it_ids, it_lg = decode_incremental(gen, vlm_bf16, use_cuda=False, forced=forced)
    ic_ids, ic_lg = decode_incremental(gen, vlm_bf16, use_cuda=True, forced=forced)
    f32_ids, f32_lg = decode_full(gen32, vlm_fp32, forced=forced)

    print("\n############ TEST 1: TEACHER-FORCED (isolates the operator) ############")
    print("-- per-position argmax agreement vs full-recompute (same prefix) --")
    for name, lg in [("incremental-torch", it_lg), ("incremental-CUDA", ic_lg)]:
        am = (lg.argmax(-1) == ff_lg.argmax(-1)).float().mean().item() * 100
        tag = "IDENTICAL" if am == 100.0 else "see logit err below"
        print(f"  {name:20s} vs full-bf16 : {am:6.2f}% argmax match   {tag}")
    am32 = (ff_lg.argmax(-1) == f32_lg.argmax(-1)).float().mean().item() * 100
    print(f"  {'full-bf16':20s} vs full-fp32 : {am32:6.2f}% argmax match   (bf16 round-off floor)")

    print("\n-- per-position LOGIT error (256 positions, teacher-forced) --")
    print(f"  {'pair':36s}   abs_max     abs_mean    rel_mean")
    for name, lg, ref in [
        ("incremental-torch vs full-bf16", it_lg, ff_lg),
        ("incremental-CUDA  vs full-bf16", ic_lg, ff_lg),
        ("incremental-CUDA  vs full-fp32", ic_lg, f32_lg),
        ("full-bf16         vs full-fp32", ff_lg, f32_lg),
    ]:
        am, ame, rm, rme = err(lg, ref)
        print(f"  {name:36s}  {am:9.3e}  {ame:9.3e}  {rme:9.3e}")
    print("  ^ if incremental-vs-bf16 error <= bf16-vs-fp32 error, the operator is")
    print("    equivalent within round-off (the divergence is precision, not logic).")

    # ============================================================
    # TEST 2 (context): FREE-RUNNING -- shows AR chaos. Same operator, but argmax
    # feedback makes any round-off snowball; even bf16-vs-fp32 sequences diverge.
    # This is EXPECTED and is NOT an operator bug -- TEST 1 is the real check.
    # ============================================================
    fr_full, _ = decode_full(gen, vlm_bf16)
    fr_t, _ = decode_incremental(gen, vlm_bf16, use_cuda=False)
    fr_c, _ = decode_incremental(gen, vlm_bf16, use_cuda=True)
    fr_f32, _ = decode_full(gen32, vlm_fp32)
    print("\n############ TEST 2: FREE-RUNNING (AR chaos, context only) ############")
    print("  (argmax feedback amplifies round-off; sequence divergence here is")
    print("   precision chaos, identical in kind to bf16-vs-fp32, NOT a path bug)")
    for name, ids, ref in [
        ("incremental-torch vs full-bf16", fr_t, fr_full),
        ("incremental-CUDA  vs full-bf16", fr_c, fr_full),
        ("full-bf16         vs full-fp32", fr_full, fr_f32),
    ]:
        pct, first = seq_match(ids, ref)
        print(f"  {name:36s}: {pct:6.2f}% seq match  (1st div @tok{first})")

