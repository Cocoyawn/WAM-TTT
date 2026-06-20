"""FINAL root-cause: compare the two FULL self-consistent eval paths end-to-end.

predict_action feeds the action head with gen_hidden_states produced by EITHER:
  (FULL) full-recompute: argmax AR loop, then forward(gen_input) -> hidden
  (INCR) generate_incremental: one O(n) pass -> (tokens, hidden) together
Each path is internally self-consistent (its tokens & hidden come from the same
forward logic). The decisive earlier test showed: GIVEN THE SAME tokens, the two
hidden -> action are equivalent (~1%). So if the FULL self-consistent action and
the INCR self-consistent action also match, incremental is usable; if they differ
a lot, the divergent token sequence (chaos) genuinely changes the action -> the
issue is intrinsic to free-running incremental, not a code bug.

This is exactly what real eval does, minus Qwen/sim (gen_context + ah vlm cond
synthesized). Reports |action_FULL - action_INCR| against scale references.
"""
import torch
from src.models.generator import ImageGeneratorTransformer
from src.models.policies import ActionDiffusionTransformerMoE

CKPT = ("/mnt/afs-h200/yuyangcheng/workplace/VLANeXt/VLANeXt_ablation_wm/"
        "ttt_chunk256_libero_mixed_clean/checkpoint_final.pt")
DEV = "cuda"
VOCAB, VLM_H, GEN_H, DEPTH, GHEADS = 131072, 2048, 768, 29, 12
PHID, PDEPTH, PHEADS = 1024, 29, 16
CHUNK, MIX, N, T_CTX, ACT_DIM = 256, 4, 256, 16, 7


def load():
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)["model_state_dict"]
    sd = {k[k.index(".") + 1:] if k.startswith("module.") else k: v for k, v in sd.items()}
    gen = ImageGeneratorTransformer(
        vocab_size=VOCAB, vlm_hidden_size=VLM_H, hidden_size=GEN_H, depth=DEPTH,
        num_heads=GHEADS, max_seq_len=1024, mixer_type="ttt", mix_every_n=MIX,
        ttt_chunk_size=CHUNK, fallback_mixer="attention", ttt_use_cuda_kernel=False)
    gen.load_state_dict({k[len("generator."):]: v for k, v in sd.items()
                         if k.startswith("generator.")}, strict=False)
    gen = gen.to(DEV, torch.bfloat16).eval()
    ah = ActionDiffusionTransformerMoE(
        action_dim=ACT_DIM, vlm_hidden_size=VLM_H, hidden_size=PHID, depth=PDEPTH,
        num_heads=PHEADS, gen_hidden_size=GEN_H, mixer_type="ttt", mix_every_n=MIX)
    ah.load_state_dict({k[len("action_head."):]: v for k, v in sd.items()
                        if k.startswith("action_head.")}, strict=False)
    ah = ah.to(DEV, torch.bfloat16).eval()
    return gen, ah


@torch.no_grad()
def full_path(gen, vlm):
    curr = torch.zeros((1, 1), dtype=torch.long, device=DEV)
    for _ in range(N):
        lg, _ = gen(curr, vlm)
        curr = torch.cat([curr, torch.argmax(lg[:, -1:], dim=-1)], dim=1)
    _, hs = gen(curr[:, :-1], vlm)
    return curr[:, 1:N + 1], hs


if __name__ == "__main__":
    torch.manual_seed(0)
    gen, ah = load()
    gs = torch.Generator(device=DEV).manual_seed(7)
    vlm = [torch.randn(1, T_CTX, VLM_H, generator=gs, device=DEV, dtype=torch.bfloat16)
           for _ in range(DEPTH)]

    ids_full, hs_full = full_path(gen, vlm)
    ids_incr, hs_incr = gen.generate_incremental(vlm, N)

    tok_match = float((ids_full == ids_incr).float().mean() * 100)
    print(f"\ntoken-seq match (full vs incr, free-run): {tok_match:.1f}%")

    # action head: same fixed action-side inputs, only gen_hidden differs by path
    noisy = torch.randn(1, 8, ACT_DIM, generator=gs, device=DEV, dtype=torch.bfloat16)
    ah_vlm = [torch.randn(1, T_CTX, VLM_H, generator=gs, device=DEV, dtype=torch.bfloat16)
              for _ in range(PDEPTH)]
    t0 = torch.zeros(1, device=DEV)
    with torch.no_grad():
        aF = ah(noisy, t0, ah_vlm, gen_hidden_states=hs_full)
        aI = ah(noisy, t0, ah_vlm, gen_hidden_states=hs_incr)
        a0 = ah(noisy, t0, ah_vlm, gen_hidden_states=None)

    dFI = (aF - aI).abs()
    dF0 = (aF - a0).abs()
    print(f"\naction:  |aFULL - aINCR| mean={float(dFI.mean()):.4e} max={float(dFI.max()):.4e}")
    print(f"         |aFULL|         mean={float(aF.abs().mean()):.4e}")
    print(f"         |aFULL - a_noGen| mean={float(dF0.mean()):.4e}  (gen-conditioning magnitude)")
    print(f"  ratio  (FULL-vs-INCR)/(FULL-vs-noGen) = {float(dFI.mean()/(dF0.mean()+1e-9)):.3f}")
    print("\nINTERPRET: small ratio -> divergent tokens still give ~same action "
          "(incremental usable); large ratio -> chaos genuinely changes the action.")
