"""
Verify two claims about TTT inference at the deployment config (chunk_size=256):

CLAIM 1 (per-position independence): with chunk_size >= seq_len, the causal TTT
op does apply-then-update as ONE chunk, so apply uses only the VLM-pre-updated
weights => output[:, j] depends only on q_j and w_vlm, NOT on other positions'
k/v. Test: perturb position p's k/v, check outputs at j != p are unchanged.

CLAIM 2 (time split): in one full-prefix generator forward, how much time is in
the 7 TTT layers vs the 22 attention layers.

Run:
  TORCHDYNAMO_DISABLE=1 PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES=<x> \
    /mnt/afs-h200/yuyangcheng/venvs/fla_triton32/bin/python -m src.models.test_infer_claims
"""
import time
import torch
from src.models.ttt import causal_block_fast_weight_swish_glu

dev = "cuda"


def claim1_independence():
    print("=== CLAIM 1: chunk=256 per-position independence ===")
    B, L, d, dh, T = 4, 256, 64, 64, 16
    g = torch.Generator(device=dev).manual_seed(0)
    rk = lambda *s: torch.randn(*s, generator=g, device=dev, dtype=torch.float32)
    pr = lambda *s: torch.rand(*s, generator=g, device=dev, dtype=torch.float32) * 0.02 + 0.001
    w0, w1, w2 = rk(B, d, dh), rk(B, dh, d), rk(B, d, dh)
    q, k, v = rk(B, L, d), rk(B, L, d), rk(B, L, d)
    lr0, lr1, lr2 = pr(B, L, 1), pr(B, L, 1), pr(B, L, 1)
    vk, vv = rk(B, T, d), rk(B, T, d)
    vl0, vl1, vl2 = pr(B, T, 1), pr(B, T, 1), pr(B, T, 1)
    args = dict(chunk_size=256, muon_update_steps=0, vlm_k=vk, vlm_v=vv,
                vlm_lr0=vl0, vlm_lr1=vl1, vlm_lr2=vl2)
    o0 = causal_block_fast_weight_swish_glu(w0.clone(), w1.clone(), w2.clone(),
                                            q, k, v, lr0, lr1, lr2, **args)[0]
    p = 100
    k2, v2 = k.clone(), v.clone()
    k2[:, p] += 5.0; v2[:, p] += 5.0
    o1 = causal_block_fast_weight_swish_glu(w0.clone(), w1.clone(), w2.clone(),
                                            q, k2, v2, lr0, lr1, lr2, **args)[0]
    other0 = torch.cat([o0[:, :p], o0[:, p + 1:]], 1)
    other1 = torch.cat([o1[:, :p], o1[:, p + 1:]], 1)
    d_other = (other1 - other0).abs().max().item()
    print(f"  perturb k/v at p={p}: max delta at OTHER positions = {d_other:.2e}")
    verdict = ("INDEPENDENT -> claim1 TRUE: chunk=256 has NO cross-position state"
               if d_other < 1e-5 else "coupled -> claim1 FALSE")
    print(f"  => {verdict}")
    q3 = q.clone(); q3[:, p] += 5.0
    o2 = causal_block_fast_weight_swish_glu(w0.clone(), w1.clone(), w2.clone(),
                                            q3, k, v, lr0, lr1, lr2, **args)[0]
    d_p = (o2[:, p] - o0[:, p]).abs().max().item()
    d_oth = (torch.cat([o2[:, :p], o2[:, p + 1:]], 1) - other0).abs().max().item()
    print(f"  perturb q at p: delta@p={d_p:.2e} (want>0), delta elsewhere={d_oth:.2e} (want~0)")


def claim2_timesplit():
    print("\n=== CLAIM 2: TTT vs attention full-prefix(256) forward time ===")
    from src.models.generator import ImageGeneratorTransformer
    for mixer in ["attention", "ttt"]:
        torch.manual_seed(0)
        gen = ImageGeneratorTransformer(
            vocab_size=1024, vlm_hidden_size=512, hidden_size=768, depth=29,
            num_heads=12, mixer_type=mixer, mix_every_n=4, ttt_chunk_size=256,
        ).to(dev, torch.bfloat16).eval()
        ids = torch.randint(0, 1024, (1, 256), device=dev)
        vlm = [torch.randn(1, 16, 512, device=dev, dtype=torch.bfloat16) for _ in range(29)]
        with torch.no_grad():
            for _ in range(3):
                gen(ids, vlm)
            torch.cuda.synchronize(); t = time.time()
            for _ in range(20):
                gen(ids, vlm)
            torch.cuda.synchronize()
        ms = (time.time() - t) / 20 * 1000
        n_ttt = sum(1 for b in gen.blocks if b.mixer_type == "ttt")
        print(f"  [{mixer:9s}] {ms:7.2f} ms | {n_ttt} TTT + {29 - n_ttt} attn layers")
        del gen; torch.cuda.empty_cache()


if __name__ == "__main__":
    claim1_independence()
    claim2_timesplit()
