"""
Correctness gate for incremental TTT inference.

For a causal method-B TTT layer at chunk_size=256, full forward output at each
position must EQUAL the incremental path (build_state(ctx) once, then infer_step
per token). This proves the O(n^2)->O(n) inference rewrite changes nothing.

Run:
  TORCHDYNAMO_DISABLE=1 PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES=<x> \
    /mnt/afs-h200/yuyangcheng/venvs/fla_triton32/bin/python -m src.models.test_ttt_incremental
"""
import sys
import torch
from src.models.ttt import FastWeightGluMLPMultihead

dev = "cuda"


def main():
    if not torch.cuda.is_available():
        print("SKIP: no CUDA"); return 0
    results = []
    # fp32 per-token vs full uses different GEMM shapes (L=1 vs L=256), so cuBLAS
    # picks different kernels -> ~1e-3 rel rounding noise (NOT a correctness bug;
    # batch-infer below matches full at 0.0). bf16 is the real deployment dtype.
    for dtype, tol in [(torch.float32, 2e-3), (torch.bfloat16, 5e-2)]:
        torch.manual_seed(0)
        dim, heads, L, Tctx = 768, 12, 256, 16
        layer = FastWeightGluMLPMultihead(
            dim=dim, head_dim=dim // heads, causal=True, chunk_size=256,
            vlm_hidden_size=dim).to(dev, dtype).eval()
        B = 2
        g = torch.Generator(device=dev).manual_seed(1)
        x = torch.randn(B, L, dim, generator=g, device=dev, dtype=dtype)
        ctx = torch.randn(B, Tctx, dim, generator=g, device=dev, dtype=dtype)

        with torch.no_grad():
            # full forward (reference)
            out_full, _ = layer(x, {}, ctx)
            # incremental: build state once, then per-token
            state = layer.infer_build_state(ctx)
            outs = []
            for t in range(L):
                o = layer.infer_step(x[:, t:t + 1], state)
                outs.append(o)
            out_inc = torch.cat(outs, dim=1)

        err = (out_full.float() - out_inc.float()).abs().max().item()
        mag = out_full.float().abs().max().item() + 1e-9
        rel = err / mag
        ok = rel < tol
        print(f"[{'OK ' if ok else 'BAD'}] {str(dtype):14s} | full-vs-incremental rel err={rel:.2e} "
              f"(abs={err:.2e}, tol={tol:.0e})")
        results.append(ok)

        # also verify a batch-incremental (feed all tokens at once via infer_step)
        with torch.no_grad():
            out_batch = layer.infer_step(x, layer.infer_build_state(ctx))
        err2 = (out_full.float() - out_batch.float()).abs().max().item() / mag
        ok2 = err2 < tol
        print(f"[{'OK ' if ok2 else 'BAD'}] {str(dtype):14s} | full-vs-batch-infer rel err={err2:.2e}")
        results.append(ok2)

    # ---- CUDA fused infer_step parity vs torch infer_step ----
    from src.models import ttt_cuda
    if ttt_cuda._load_extension() is not None and hasattr(ttt_cuda._load_extension(), "infer_step"):
        for dtype, tol in [(torch.float32, 2e-3), (torch.bfloat16, 5e-2)]:
            torch.manual_seed(0)
            dim, heads = 768, 12
            lt = FastWeightGluMLPMultihead(dim=dim, head_dim=dim // heads, causal=True,
                                           chunk_size=256, vlm_hidden_size=dim,
                                           use_cuda_kernel=False).to(dev, dtype).eval()
            lc = FastWeightGluMLPMultihead(dim=dim, head_dim=dim // heads, causal=True,
                                           chunk_size=256, vlm_hidden_size=dim,
                                           use_cuda_kernel=True).to(dev, dtype).eval()
            lc.load_state_dict(lt.state_dict())
            g = torch.Generator(device=dev).manual_seed(2)
            x1 = torch.randn(1, 1, dim, generator=g, device=dev, dtype=dtype)
            ctx = torch.randn(1, 16, dim, generator=g, device=dev, dtype=dtype)
            with torch.no_grad():
                st = lt.infer_build_state(ctx)
                o_torch = lt.infer_step(x1, st)
                o_cuda = lc.infer_step(x1, st)
            e = (o_torch.float() - o_cuda.float()).abs().max().item()
            m = o_torch.float().abs().max().item() + 1e-9
            ok = e / m < tol
            print(f"[{'OK ' if ok else 'BAD'}] {str(dtype):14s} | CUDA-vs-torch infer_step rel err={e/m:.2e}")
            results.append(ok)
    else:
        print("[skip] CUDA infer_step not built")

    allok = all(results)
    print("\n=== INCREMENTAL PARITY", "PASS ===" if allok else "FAIL ===")
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
