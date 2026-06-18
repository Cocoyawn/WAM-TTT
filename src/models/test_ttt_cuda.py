"""
Numerical parity test: CUDA causal TTT vs the torch reference.

Validates `src/models/ttt_cuda` against `causal_block_fast_weight_swish_glu`
from `src/models/ttt.py`, at the REAL vision-expert shapes:
    head_dim=64, d_h=64, num_heads=12, 256 image tokens, chunk_size=256,
    with the global VLM-context pre-update (ctx present), muon_update_steps=0.

Checks:
  1. output max-abs-err vs torch reference (bf16 tolerance)
  2. updated fast-weight (w0/w1/w2) max-abs-err
  3. causality: perturbing query token p must not change outputs < p
     (only meaningful when chunk_size < L; tested with a small chunk_size too)

Run (needs a GPU; timing is NOT measured here so a shared card is fine):
    TORCHDYNAMO_DISABLE=1 PYTHONPATH="$PWD" \
      /mnt/afs-h200/yuyangcheng/venvs/fla_triton32/bin/python -m src.models.test_ttt_cuda
"""
import os
import sys

import torch

# eager reference (compile disabled keeps it simple + matches training env)
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
try:
    torch._dynamo.config.suppress_errors = True
except Exception:
    pass

from src.models.ttt import causal_block_fast_weight_swish_glu
from src.models import ttt_cuda


def _mk(B, L, d, dh, T_ctx, dtype, device, seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    rk = lambda *s: torch.randn(*s, generator=g, device=device, dtype=dtype)
    w0 = rk(B, d, dh)
    w1 = rk(B, dh, d)
    w2 = rk(B, d, dh)
    q = rk(B, L, d)
    k = rk(B, L, d)
    v = rk(B, L, d)
    # lr is fp32 and positive (softplus output in the real path)
    lr0 = torch.rand(B, L, 1, generator=g, device=device, dtype=torch.float32) * 0.02 + 0.001
    lr1 = torch.rand(B, L, 1, generator=g, device=device, dtype=torch.float32) * 0.02 + 0.001
    lr2 = torch.rand(B, L, 1, generator=g, device=device, dtype=torch.float32) * 0.02 + 0.001
    ck = rk(B, T_ctx, d)
    cv = rk(B, T_ctx, d)
    cl0 = torch.rand(B, T_ctx, 1, generator=g, device=device, dtype=torch.float32) * 0.02 + 0.001
    cl1 = torch.rand(B, T_ctx, 1, generator=g, device=device, dtype=torch.float32) * 0.02 + 0.001
    cl2 = torch.rand(B, T_ctx, 1, generator=g, device=device, dtype=torch.float32) * 0.02 + 0.001
    return dict(w0=w0, w1=w1, w2=w2, q=q, k=k, v=v, lr0=lr0, lr1=lr1, lr2=lr2,
                ck=ck, cv=cv, cl0=cl0, cl1=cl1, cl2=cl2)


def _run_ref(t, chunk_size):
    return causal_block_fast_weight_swish_glu(
        t["w0"].clone(), t["w1"].clone(), t["w2"].clone(),
        t["q"], t["k"], t["v"], t["lr0"], t["lr1"], t["lr2"],
        chunk_size=chunk_size, muon_update_steps=0,
        vlm_k=t["ck"], vlm_v=t["cv"], vlm_lr0=t["cl0"], vlm_lr1=t["cl1"], vlm_lr2=t["cl2"],
    )


def _run_cuda(t, chunk_size):
    return ttt_cuda.causal_ttt_forward(
        t["w0"].clone(), t["w1"].clone(), t["w2"].clone(),
        t["q"], t["k"], t["v"], t["lr0"], t["lr1"], t["lr2"],
        chunk_size=chunk_size,
        vlm_k=t["ck"], vlm_v=t["cv"], vlm_lr0=t["cl0"], vlm_lr1=t["cl1"], vlm_lr2=t["cl2"],
    )


def _err(a, b):
    return (a.float() - b.float()).abs().max().item()


def _rel_err(a, b):
    # max abs err normalized by the reference's max abs magnitude
    denom = a.float().abs().max().item() + 1e-9
    return (a.float() - b.float()).abs().max().item() / denom


def main():
    if not torch.cuda.is_available():
        print("SKIP: no CUDA device")
        return 0
    dev = "cuda"
    # bf16: the real training/inference dtype; fp32 also tested for tighter bound.
    results = []
    # ---- fp32: strict absolute/relative parity (the math must be correct) ----
    for chunk_size in (256, 64):  # 256 = real config (1 chunk); 64 = multi-chunk causal
        t = _mk(B=24, L=256, d=64, dh=64, T_ctx=32, dtype=torch.float32, device=dev, seed=1)
        o_ref, w0r, w1r, w2r = _run_ref(t, chunk_size)
        try:
            o_c, w0c, w1c, w2c = _run_cuda(t, chunk_size)
        except RuntimeError as e:
            print(f"FAIL build/run (fp32, cs={chunk_size}): {e}")
            return 1
        eo = _rel_err(o_ref, o_c)
        ew = max(_err(w0r, w0c), _err(w1r, w1c), _err(w2r, w2c))
        tol = 2e-3
        ok = eo < tol and ew < tol
        print(f"[{'OK ' if ok else 'BAD'}] fp32          cs={chunk_size:3d} | "
              f"out_relerr={eo:.2e} w_err={ew:.2e} (tol={tol:.0e})")
        results.append(ok)

    # ---- bf16: the reference is ITSELF only bf16-accurate, so an absolute tol
    # is meaningless. Fair criterion: CUDA-bf16 must be no less accurate than
    # torch-bf16, both measured against the fp32 ground truth. Allow 1.5x slack.
    for chunk_size in (256, 64):
        # same bf16 inputs for all three; fp32 truth = those inputs upcast.
        t16 = _mk(B=24, L=256, d=64, dh=64, T_ctx=32, dtype=torch.bfloat16, device=dev, seed=1)
        t32 = {kk: (vv.float() if torch.is_tensor(vv) and vv.dtype == torch.bfloat16 else vv)
               for kk, vv in t16.items()}
        o_truth, w0t, w1t, w2t = _run_ref(t32, chunk_size)     # fp32 ground truth
        o_tref, w0tr, w1tr, w2tr = _run_ref(t16, chunk_size)   # torch bf16
        o_cu, w0cu, w1cu, w2cu = _run_cuda(t16, chunk_size)    # cuda bf16
        # output relative error vs fp32 truth
        ref_oerr = _rel_err(o_truth, o_tref)
        cu_oerr = _rel_err(o_truth, o_cu)
        # weight error vs fp32 truth
        ref_werr = max(_err(w0t, w0tr), _err(w1t, w1tr), _err(w2t, w2tr))
        cu_werr = max(_err(w0t, w0cu), _err(w1t, w1cu), _err(w2t, w2cu))
        ok = cu_oerr <= ref_oerr * 1.5 + 1e-6 and cu_werr <= ref_werr * 1.5 + 1e-6
        print(f"[{'OK ' if ok else 'BAD'}] bf16          cs={chunk_size:3d} | "
              f"out: cuda={cu_oerr:.2e} vs torch={ref_oerr:.2e} | "
              f"w: cuda={cu_werr:.2e} vs torch={ref_werr:.2e} (cuda<=1.5x torch)")
        results.append(ok)

    # causality (fp32, small chunk so it's a real test): perturb q token p,
    # outputs at positions < p (in earlier chunks) must be unchanged.
    t = _mk(B=4, L=128, d=64, dh=64, T_ctx=16, dtype=torch.float32, device=dev, seed=2)
    cs = 32
    o0, *_ = _run_cuda(t, cs)
    p = 100
    t2 = {kk: (vv.clone() if torch.is_tensor(vv) else vv) for kk, vv in t.items()}
    t2["q"][:, p, :] += 1.0
    t2["k"][:, p, :] += 1.0
    t2["v"][:, p, :] += 1.0
    o1, *_ = _run_cuda(t2, cs)
    # positions strictly before p's chunk start must be bitwise-ish unchanged
    chunk_start = (p // cs) * cs
    delta_before = (o0[:, :chunk_start] - o1[:, :chunk_start]).abs().max().item()
    delta_after = (o0[:, p:] - o1[:, p:]).abs().max().item()
    causal_ok = delta_before < 1e-4 and delta_after > 1e-3
    print(f"[{'OK ' if causal_ok else 'BAD'}] causality | "
          f"before_p_delta={delta_before:.2e} (want~0) after_p_delta={delta_after:.2e} (want>0)")
    results.append(causal_ok)

    allok = all(results)
    print("\n=== PARITY", "PASS ===" if allok else "FAIL ===")
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
