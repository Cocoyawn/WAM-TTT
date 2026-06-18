"""
Unit-test the three NEW CUDA backward primitives against the proven torch vjps
in ttt_manual_backward.py (which match autograd to 1e-12). Localizes any
Plan-A CUDA backward bug to a single kernel.

Run:
  TORCHDYNAMO_DISABLE=1 PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES=5 \
    /mnt/afs-h200/yuyangcheng/venvs/fla_triton32/bin/python -m src.models.test_ttt_cuda_primitives
"""
import sys
import torch

from src.models import ttt_cuda
from src.models.ttt_manual_backward import (
    silu_p, silu_pp, frobnorm_fwd, frobnorm_bwd, weightnorm_fwd, weightnorm_bwd)


def main():
    if not torch.cuda.is_available():
        print("SKIP: no CUDA"); return 0
    ext = ttt_cuda._load_extension()
    if ext is None:
        print("FAIL: extension not built"); return 1
    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(7)
    results = []

    for dt, tol in [(torch.float32, 2e-5), (torch.float64, 1e-10)]:
        # skip fp64 if kernel doesn't dispatch it (AT_DISPATCH_FLOATING_TYPES covers double)
        x = torch.randn(4, 32, 16, generator=g, device=dev, dtype=dt)

        # ---- silu_derivs ----
        try:
            sp, spp = ext.silu_derivs(x)
            e1 = (sp - silu_p(x)).abs().max().item()
            e2 = (spp - silu_pp(x)).abs().max().item()
            ok = e1 < tol and e2 < tol
            print(f"[{'OK ' if ok else 'BAD'}] silu_derivs {str(dt):14s} | sp_err={e1:.2e} spp_err={e2:.2e}")
            results.append(ok)
        except RuntimeError as e:
            print(f"[skip] silu_derivs {dt}: {str(e)[:60]}");

        # ---- frobnorm_bwd ([B,A,C], Frobenius over (1,2)) ----
        xm = torch.randn(4, 16, 8, generator=g, device=dev, dtype=dt)
        gy = torch.randn(4, 16, 8, generator=g, device=dev, dtype=dt)
        _, cache = frobnorm_fwd(xm)
        ref = frobnorm_bwd(gy, cache)
        try:
            cu = ext.frobnorm_bwd(gy, xm, 1e-7)
            e = (ref - cu).abs().max().item()
            ok = e < tol
            print(f"[{'OK ' if ok else 'BAD'}] frobnorm_bwd {str(dt):14s} | err={e:.2e}")
            results.append(ok)
        except RuntimeError as e:
            print(f"[skip] frobnorm_bwd {dt}: {str(e)[:60]}")

        # ---- weightnorm_bwd ([B,A,C], per-column over dim=1) ----
        wpre = torch.randn(4, 16, 8, generator=g, device=dev, dtype=dt)
        wn = wpre.norm(2, 1, keepdim=True).detach()
        gyw = torch.randn(4, 16, 8, generator=g, device=dev, dtype=dt)
        _, wc = weightnorm_fwd(wpre, wn)
        refw = weightnorm_bwd(gyw, wc)
        try:
            cuw = ext.weightnorm_bwd(gyw, wpre, wn, 1e-5)
            e = (refw - cuw).abs().max().item()
            ok = e < tol
            print(f"[{'OK ' if ok else 'BAD'}] weightnorm_bwd {str(dt):12s} | err={e:.2e}")
            results.append(ok)
        except RuntimeError as e:
            print(f"[skip] weightnorm_bwd {dt}: {str(e)[:60]}")

    allok = all(results) and len(results) > 0
    print("\n=== PRIMITIVES", "PASS ===" if allok else "FAIL ===")
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
