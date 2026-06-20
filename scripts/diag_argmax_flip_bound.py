"""Quantify the argmax-flip bound: how low must the incremental hidden relative
error be to reproduce the full-recompute token sequence (seamless drop-in)?

Along the full-recompute free-run trajectory we measure, per generation step:
  - margin m_t = top1_logit - top2_logit   (the gap argmax must preserve)
  - logit scale, and the empirical perturbation the incremental path injects
    into the logits at that step (teacher-forced on the SAME prefix, so it's the
    pure round-off perturbation, not chaos).
A step survives if injected logit perturbation < m_t / 2 (sufficient condition).
The sequence survives iff EVERY step survives -> bound set by min_t m_t and the
worst-case perturbation. We also fit perturbation vs hidden-rel-error to back out
the required hidden-rel-error threshold.

Reports: margin distribution (min/p1/p10/median), current per-step logit
perturbation distribution, fraction of steps that already flip, and the implied
hidden-rel-error target.
"""
import json, torch
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
        ttt_chunk_size=CHUNK, fallback_mixer="attention", ttt_use_cuda_kernel=False)
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)["model_state_dict"]
    g.load_state_dict({k[k.index("generator.") + 10:]: v for k, v in sd.items()
                       if "generator." in k}, strict=False)
    return g.to(DEV, dtype).eval()


@torch.no_grad()
def full_freerun_tokens(g, vlm):
    curr = torch.zeros((1, 1), dtype=torch.long, device=DEV)
    for _ in range(N):
        lg, _ = g(curr, vlm)
        curr = torch.cat([curr, torch.argmax(lg[:, -1:], dim=-1)], dim=1)
    return curr  # [1, N+1]  (BOS + N tokens)


@torch.no_grad()
def incr_logits_teacher_forced(g, vlm, forced):
    """Per-step logits from the incremental path, teacher-forced on `forced`
    (same prefix as full-recompute) -> isolates the round-off perturbation."""
    blocks = g.blocks
    relevant = vlm[-len(blocks):]
    dt = next(g.parameters()).dtype
    states = [b.infer_init(v.to(dtype=dt)) for b, v in zip(blocks, relevant)]
    out = []
    for t in range(forced.shape[1]):
        x = g.token_emb(forced[:, t:t + 1]) + g.pos_embed[:, t:t + 1, :]
        for b, st in zip(blocks, states):
            x = b.infer_step(x, st)
        out.append(g.head(g.norm_final(x))[:, -1, :].float())  # (1,V)
    return torch.cat(out, dim=0)  # (N, V)


@torch.no_grad()
def full_logits_teacher_forced(g, vlm, forced):
    lg, _ = g(forced, vlm)
    return lg[0].float()  # (N, V)


def pct(x, p):
    return float(torch.quantile(x, p))


if __name__ == "__main__":
    g = build(torch.bfloat16)
    gs = torch.Generator(device=DEV).manual_seed(7)
    vlm = [torch.randn(1, T_CTX, VLM_H, generator=gs, device=DEV, dtype=torch.bfloat16)
           for _ in range(DEPTH)]

    curr = full_freerun_tokens(g, vlm)
    forced = curr[:, :N]                       # positions fed at each step
    lg_full = full_logits_teacher_forced(g, vlm, forced)   # (N,V) ground-truth logits
    lg_incr = incr_logits_teacher_forced(g, vlm, forced)   # (N,V) incremental logits

    # margins from full-recompute logits
    top2 = torch.topk(lg_full, 2, dim=-1).values            # (N,2)
    margin = (top2[:, 0] - top2[:, 1])                      # (N,)
    # per-step logit perturbation injected by incremental (at the argmax token)
    # use max-abs over vocab as worst-case, and the gap-relevant delta
    dperturb = (lg_incr - lg_full).abs().amax(dim=-1)       # (N,) worst-case per step
    # does incremental's argmax differ from full's, per step (teacher-forced)?
    flip = (lg_incr.argmax(-1) != lg_full.argmax(-1)).float()

    # hidden rel error at final layer (teacher-forced) ~ 2.1% known; ratio to logit
    print(f"\n== margin m_t = top1-top2 (full-recompute logits, {N} steps) ==")
    print(f"  min={margin.min():.4f}  p1={pct(margin,0.01):.4f}  p10={pct(margin,0.10):.4f}  "
          f"median={pct(margin,0.5):.4f}  max={margin.max():.4f}")
    print(f"\n== incremental per-step logit perturbation (worst-case over vocab) ==")
    print(f"  median={pct(dperturb,0.5):.4f}  p90={pct(dperturb,0.9):.4f}  max={dperturb.max():.4f}")
    print(f"\n== teacher-forced per-step argmax flip rate ==")
    print(f"  flips = {int(flip.sum())}/{N}  ({100*flip.mean():.1f}%)")
    # survival condition: perturb < margin/2 per step
    safe = (dperturb < margin / 2).float()
    print(f"  steps satisfying perturb<m/2 = {int(safe.sum())}/{N}")
    # bound: to survive ALL steps, need perturb < min_t(margin_t)/2
    need_logit = float(margin.min() / 2)
    cur_logit = float(dperturb.median())
    print(f"\n== bound ==")
    print(f"  worst-step margin/2 = {need_logit:.4f}  (logit perturb must stay below this)")
    print(f"  current median logit perturb = {cur_logit:.4f}")
    print(f"  => must shrink logit perturb by ~{cur_logit/max(need_logit,1e-6):.1f}x")
    print(f"  current hidden rel-err (final layer, teacher-forced) ~2.1%")
    print(f"  => target hidden rel-err ~ {2.1 * need_logit/max(cur_logit,1e-6):.4f}%  "
          f"(linear extrapolation)")
    json.dump({"margin_min": float(margin.min()), "margin_p1": pct(margin,0.01),
               "margin_p10": pct(margin,0.10), "margin_median": pct(margin,0.5),
               "perturb_median": pct(dperturb,0.5), "perturb_max": float(dperturb.max()),
               "tf_flip_rate": float(flip.mean()), "N": N},
              open("/mnt/afs-h200/yuyangcheng/workplace/VLANeXt/docs/argmax_flip_bound.json","w"),
              indent=2)
    print("\nwrote docs/argmax_flip_bound.json")
