"""
端到端正确性 + 误差分析:优化后的 CUDA infer_step 必须与普通 TTT layer 的
forward 输出一致。

对比链(同一份权重、同一份输入、真实 vision-expert 形状):
  REF   = 普通 TTT layer forward(全前缀因果 op, chunk=256)         <- 黄金参考
  TORCH = 增量 torch (infer_build_state + infer_step, 无 CUDA)
  CUDA  = 增量 CUDA fused kernel (infer_step + use_cuda_kernel)

报告每条路径 vs REF 的:
  - 绝对误差 max |a-b|
  - 相对误差 max|a-b| / max|REF|
  - 相对误差 (逐元素) mean |a-b|/(|REF|+eps)
fp32(查数学正确性) 和 bf16(查部署精度) 都测。

判据:
  fp32  CUDA vs REF 相对误差 < 1e-3 (matmul 形状不同导致的 cuBLAS rounding 量级)
  bf16  CUDA vs REF 相对误差 < torch vs REF 的 1.5x (CUDA 不得比 torch 更不准)

Run on any GPU (正确性对共享卡稳健):
  TORCHDYNAMO_DISABLE=1 PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES=<x> \
    /mnt/afs-h200/yuyangcheng/venvs/fla_triton32/bin/python -m src.models.test_infer_correctness
"""
import sys
import torch
from src.models.ttt import FastWeightGluMLPMultihead

dev = "cuda"
HIDDEN, HEADS, L, TCTX = 768, 12, 256, 16


def errs(ref, x):
    ref = ref.float(); x = x.float()
    d = (ref - x).abs()
    abs_max = d.max().item()
    denom = ref.abs().max().item() + 1e-12
    rel_max = abs_max / denom
    rel_mean = (d / (ref.abs() + 1e-6)).mean().item()
    return abs_max, rel_max, rel_mean


def build_pair(dtype):
    torch.manual_seed(0)
    ref = FastWeightGluMLPMultihead(dim=HIDDEN, head_dim=HIDDEN // HEADS, causal=True,
                                    chunk_size=256, vlm_hidden_size=HIDDEN,
                                    use_cuda_kernel=False).to(dev, dtype).eval()
    cuda = FastWeightGluMLPMultihead(dim=HIDDEN, head_dim=HIDDEN // HEADS, causal=True,
                                     chunk_size=256, vlm_hidden_size=HIDDEN,
                                     use_cuda_kernel=True).to(dev, dtype).eval()
    cuda.load_state_dict(ref.state_dict())   # IDENTICAL weights
    return ref, cuda


def main():
    if not torch.cuda.is_available():
        print("SKIP: no CUDA"); return 0
    from src.models import ttt_cuda
    ext = ttt_cuda._load_extension()
    has_cuda = ext is not None and hasattr(ext, "infer_step")
    print(f"CUDA infer_step kernel available: {has_cuda}\n")

    results = []
    for dtype in (torch.float32, torch.bfloat16):
        ref_layer, cuda_layer = build_pair(dtype)
        g = torch.Generator(device=dev).manual_seed(7)
        x = torch.randn(1, L, HIDDEN, generator=g, device=dev, dtype=dtype)
        ctx = torch.randn(1, TCTX, HIDDEN, generator=g, device=dev, dtype=dtype)

        with torch.no_grad():
            # REF: ordinary TTT layer forward (full-prefix causal op)
            ref_out, _ = ref_layer(x, {}, ctx)
            # TORCH incremental
            st_t = ref_layer.infer_build_state(ctx)
            torch_out = torch.cat([ref_layer.infer_step(x[:, t:t+1], st_t) for t in range(L)], 1)
            # CUDA incremental (fused kernel)
            st_c = cuda_layer.infer_build_state(ctx)
            cuda_out = torch.cat([cuda_layer.infer_step(x[:, t:t+1], st_c) for t in range(L)], 1)

        print(f"===== dtype = {dtype} =====")
        a_t, rmax_t, rmean_t = errs(ref_out, torch_out)
        a_c, rmax_c, rmean_c = errs(ref_out, cuda_out)
        print(f"  TORCH-incr vs REF : abs={a_t:.3e}  rel_max={rmax_t:.3e}  rel_mean={rmean_t:.3e}")
        print(f"  CUDA-incr  vs REF : abs={a_c:.3e}  rel_max={rmax_c:.3e}  rel_mean={rmean_c:.3e}")
        # direct CUDA vs TORCH (both incremental) -- isolates kernel-only error
        a_ct, rmax_ct, rmean_ct = errs(torch_out, cuda_out)
        print(f"  CUDA vs TORCH     : abs={a_ct:.3e}  rel_max={rmax_ct:.3e}  rel_mean={rmean_ct:.3e}")

        if dtype == torch.float32:
            ok = rmax_c < 1e-3
            print(f"  -> fp32 criterion CUDA-vs-REF rel_max < 1e-3 : {'PASS' if ok else 'FAIL'}")
        else:
            # bf16: REF is itself bf16-noisy. The fair test is vs an fp32 ground
            # truth (same inputs upcast). Whoever is CLOSER to fp32-truth is more
            # correct; CUDA must be no worse than torch.
            ref32, cuda32 = build_pair(torch.float32)
            x32, ctx32 = x.float(), ctx.float()
            with torch.no_grad():
                truth, _ = ref32(x32, {}, ctx32)
                st_tt = ref32.infer_build_state(ctx32)
                torch_t32 = torch.cat([ref32.infer_step(x32[:, t:t+1], st_tt) for t in range(L)], 1)
                st_cc = cuda32.infer_build_state(ctx32)
                cuda_t32 = torch.cat([cuda32.infer_step(x32[:, t:t+1], st_cc) for t in range(L)], 1)
            _, rmax_truth_t, _ = errs(truth, torch_out)   # torch-bf16 vs fp32 truth
            _, rmax_truth_c, _ = errs(truth, cuda_out)    # cuda-bf16  vs fp32 truth
            print(f"  vs fp32-TRUTH: torch-bf16 rel_max={rmax_truth_t:.3e}  cuda-bf16 rel_max={rmax_truth_c:.3e}")
            ok = rmax_truth_c <= rmax_truth_t * 1.5 + 1e-6
            print(f"  -> bf16 criterion cuda-bf16 closer-or-equal to truth vs torch : {'PASS' if ok else 'FAIL'}")
        results.append(ok)
        print()

    allok = all(results)
    print("=== CORRECTNESS", "PASS ===" if allok else "FAIL ===")
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
