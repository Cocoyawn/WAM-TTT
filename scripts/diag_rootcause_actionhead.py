"""DECISIVE root-cause experiment.

Question: incremental eval gives SR=0 while torch eval is fine. Is it because
  (H1) the token SEQUENCE diverges (chaos) -> different but valid images, OR
  (H2) the gen_hidden_states PRODUCTION itself differs structurally between
       incremental and full-recompute, feeding the action head bad conditioning?

Test: load generator + the diffusion action head from the REAL checkpoint.
Pick ONE fixed gen_context and ONE fixed token sequence. Produce gen_hidden_states
TWO ways on the SAME tokens:
  (A) full-recompute: generator.forward(fixed_tokens)         [training/eval path]
  (B) incremental:    per-token infer_step on fixed_tokens     [generate_incremental]
Feed BOTH into the action head (same noisy_action, same vlm cond, fixed timestep)
and compare the predicted action.

  - If action(A) ~= action(B): gen_hidden_states production is equivalent -> the
    SR=0 is purely token-sequence chaos (H1). Then the fix is about token seq.
  - If action(A) != action(B) on the SAME tokens: H2, structural bug in how
    incremental builds gen_hidden_states. That is the real, fixable root cause.

Also report per-layer hidden |Delta| (already known ~<=2.2%) for context.
No Qwen, no sim: gen_context + action-head vlm cond are synthesized at correct shape.
"""
import torch
from src.models.generator import ImageGeneratorTransformer
from src.models.policies import ActionDiffusionTransformerMoE

CKPT = ("/mnt/afs-h200/yuyangcheng/workplace/VLANeXt/VLANeXt_ablation_wm/"
        "ttt_chunk256_libero_mixed_clean/checkpoint_final.pt")
DEV = "cuda"
VOCAB, VLM_H, GEN_H, DEPTH, GHEADS = 131072, 2048, 768, 29, 12
PHID, PDEPTH, PHEADS = 1024, 29, 16  # action head (policy)
CHUNK, MIX, N, T_CTX = 256, 4, 256, 16
ACT_DIM = 7


def load():
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)["model_state_dict"]
    sd = {k[k.index(".") + 1:] if k.startswith("module.") else k: v for k, v in sd.items()}
    gen = ImageGeneratorTransformer(
        vocab_size=VOCAB, vlm_hidden_size=VLM_H, hidden_size=GEN_H, depth=DEPTH,
        num_heads=GHEADS, max_seq_len=1024, mixer_type="ttt", mix_every_n=MIX,
        ttt_chunk_size=CHUNK, fallback_mixer="attention", ttt_use_cuda_kernel=False)
    gsd = {k[len("generator."):]: v for k, v in sd.items() if k.startswith("generator.")}
    gen.load_state_dict(gsd, strict=False)
    gen = gen.to(DEV, torch.bfloat16).eval()
    return gen, sd


@torch.no_grad()
def incr_hidden_forced(gen, vlm, forced):
    blocks = gen.blocks
    relevant = vlm[-len(blocks):]
    dt = next(gen.parameters()).dtype
    states = [b.infer_init(v.to(dtype=dt)) for b, v in zip(blocks, relevant)]
    hs = [[] for _ in blocks]
    for t in range(forced.shape[1]):
        x = gen.token_emb(forced[:, t:t + 1]) + gen.pos_embed[:, t:t + 1, :]
        for li, (b, st) in enumerate(zip(blocks, states)):
            x = b.infer_step(x, st)
            hs[li].append(x)
    return [torch.cat(h, dim=1) for h in hs]


if __name__ == "__main__":
    torch.manual_seed(0)
    gen, sd = load()
    gs = torch.Generator(device=DEV).manual_seed(7)
    vlm = [torch.randn(1, T_CTX, VLM_H, generator=gs, device=DEV, dtype=torch.bfloat16)
           for _ in range(DEPTH)]

    # one fixed token sequence (full-recompute's own argmax free-run)
    curr = torch.zeros((1, 1), dtype=torch.long, device=DEV)
    for _ in range(N):
        lg, _ = gen(curr, vlm)
        curr = torch.cat([curr, torch.argmax(lg[:, -1:], dim=-1)], dim=1)
    forced = curr[:, :N]

    # (A) full-recompute hidden on the fixed seq ; (B) incremental on the SAME seq
    _, hsA = gen(forced, vlm)
    hsB = incr_hidden_forced(gen, vlm, forced)

    rels = [float((a - b).abs().mean() / b.abs().mean()) * 100 for a, b in zip(hsB, hsA)]
    print(f"\nper-layer hidden rel-err on SAME tokens: L0={rels[0]:.3f}%  "
          f"L28={rels[-1]:.3f}%  max={max(rels):.3f}%")

    # build action head, load weights
    try:
        ah = ActionDiffusionTransformerMoE(
            hidden_size=PHID, depth=PDEPTH, num_heads=PHEADS,
            vlm_hidden_size=VLM_H, gen_hidden_size=GEN_H,
            mixer_type="ttt", mix_every_n=MIX, action_dim=ACT_DIM)
    except Exception as e:
        print("action-head construct failed (signature guess):", e)
        raise SystemExit

    ahsd = {k[len("action_head."):]: v for k, v in sd.items() if k.startswith("action_head.")}
    miss, unexp = ah.load_state_dict(ahsd, strict=False)
    ah = ah.to(DEV, torch.bfloat16).eval()
    print(f"action_head loaded: missing={len(miss)} unexpected={len(unexp)}")

    # fixed action-head inputs; vary ONLY gen_hidden_states (A vs B)
    NA = 8
    noisy = torch.randn(1, NA, ACT_DIM, generator=gs, device=DEV, dtype=torch.bfloat16)
    ah_vlm = [torch.randn(1, T_CTX, VLM_H, generator=gs, device=DEV, dtype=torch.bfloat16)
              for _ in range(PDEPTH)]
    tstep = torch.zeros(1, device=DEV)

    with torch.no_grad():
        actA = ah(noisy, tstep, ah_vlm, gen_hidden_states=hsA)
        actB = ah(noisy, tstep, ah_vlm, gen_hidden_states=hsB)
        act0 = ah(noisy, tstep, ah_vlm, gen_hidden_states=None)  # no-gen reference

    dAB = (actA - actB).abs()
    dA0 = (actA - act0).abs()
    print(f"\naction-head output (same tokens, vary only gen_hidden_states A vs B):")
    print(f"  |actA - actB| mean={float(dAB.mean()):.4e} max={float(dAB.max()):.4e}")
    print(f"  ref |actA|     mean={float(actA.abs().mean()):.4e}")
    print(f"  for scale: |actA - act_noGen| mean={float(dA0.mean()):.4e}")
    print(f"  => rel(A vs B)/rel(A vs noGen) = {float(dAB.mean()/ (dA0.mean()+1e-9)):.3f}")
    print("\nINTERPRET: if |actA-actB| << |actA-act_noGen| and << |actA|, then "
          "gen_hidden_states production is equivalent -> SR=0 is token-seq chaos (H1).")
