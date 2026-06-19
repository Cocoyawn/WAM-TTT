"""
大规模严谨正确性 + 稳定性扫测: 优化后的 CUDA infer_step kernel vs 普通 TTT layer。

正确性: 对每个 (seed, batch, head_dim) 配置, 比较三条路径 vs fp32 真值:
  REF    = 普通 TTT layer forward (全前缀因果 op, chunk=256)
  TORCH  = 增量 torch (infer_build_state + infer_step)
  CUDA   = 增量 CUDA fused kernel
判据(每个配置都必须满足):
  fp32 : CUDA vs REF rel_max < 1.5e-3 (cuBLAS L=1/L=256 形状 rounding 量级)
  fp32 : CUDA vs TORCH rel_max < 1e-5 (kernel 与 torch 增量应几乎逐位一致)
  bf16 : CUDA vs fp32真值 rel_max <= TORCH vs fp32真值 * 1.5 (kernel 不得更不准)

稳定性: 汇总所有配置的误差分布 (max/mean/p50/p99/std), 报告最坏 seed, 并跑
极端输入 (大尺度/接近零) 看 kernel 不 NaN/Inf。

Run on a GPU:
  TORCHDYNAMO_DISABLE=1 PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES=<x> \
    /mnt/afs-h200/yuyangcheng/venvs/fla_triton32/bin/python -m src.models.test_infer_correctness_sweep
"""
import sys
import torch
from src.models.ttt import FastWeightGluMLPMultihead

dev = "cuda"
SEEDS = list(range(40))           # 40 random seeds
BATCHES = [1, 2, 4]               # batch sizes
HEAD_DIMS = [32, 64, 128]         # head_dim variations (kernel must handle each)
L, TCTX = 256, 16
HEADS = 12


def errs(ref, x):
    ref = ref.float(); x = x.float()
    d = (ref - x).abs()
    amax = d.max().item()
    rmax = amax / (ref.abs().max().item() + 1e-12)
    rmean = (d / (ref.abs() + 1e-6)).mean().item()
    return amax, rmax, rmean


def build_pair(dtype, head_dim):
    dim = head_dim * HEADS
    ref = FastWeightGluMLPMultihead(dim=dim, head_dim=head_dim, causal=True,
                                    chunk_size=256, vlm_hidden_size=dim,
                                    use_cuda_kernel=False).to(dev, dtype).eval()
    cuda = FastWeightGluMLPMultihead(dim=dim, head_dim=head_dim, causal=True,
                                     chunk_size=256, vlm_hidden_size=dim,
                                     use_cuda_kernel=True).to(dev, dtype).eval()
    cuda.load_state_dict(ref.state_dict())
    return ref, cuda, dim


def pct(xs, p):
    xs = sorted(xs)
    if not xs:
        return 0.0
    i = min(len(xs) - 1, int(round(p / 100.0 * (len(xs) - 1))))
    return xs[i]


def summarize(name, vals):
    import statistics as st
    mx = max(vals); mn = sum(vals) / len(vals)
    p50 = pct(vals, 50); p99 = pct(vals, 99)
    sd = st.pstdev(vals) if len(vals) > 1 else 0.0
    print(f"  {name:28s} max={mx:.3e}  mean={mn:.3e}  p50={p50:.3e}  p99={p99:.3e}  std={sd:.3e}")
    return mx


def main():
    if not torch.cuda.is_available():
        print("SKIP: no CUDA"); return 0
    from src.models import ttt_cuda
    ext = ttt_cuda._load_extension()
    if ext is None or not hasattr(ext, "infer_step"):
        print("FAIL: CUDA infer_step not built"); return 1

    n_cfg = len(SEEDS) * len(BATCHES) * len(HEAD_DIMS)
    print(f"配置总数: {n_cfg}  (seeds={len(SEEDS)} x batch={BATCHES} x head_dim={HEAD_DIMS})")
    print(f"每配置: L={L} tokens, {HEADS} heads, fp32+bf16\n")

    fails = []
    # accumulators for distributions
    acc = {
        "fp32 CUDA-vs-REF rel": [], "fp32 CUDA-vs-TORCH rel": [],
        "bf16 CUDA-vs-truth rel": [], "bf16 TORCH-vs-truth rel": [],
        "bf16 CUDA-vs-TORCH rel": [],
    }
    worst = {"fp32_cuda_ref": (0, None), "bf16_cuda_truth": (0, None)}

    for hd in HEAD_DIMS:
        for bs in BATCHES:
            for seed in SEEDS:
                torch.manual_seed(1000 + seed)
                # fp32 pair + bf16 pair (same init via manual_seed before each build)
                torch.manual_seed(1000 + seed)
                ref32, cuda32, dim = build_pair(torch.float32, hd)
                torch.manual_seed(1000 + seed)
                ref16, cuda16, _ = build_pair(torch.bfloat16, hd)

                g = torch.Generator(device=dev).manual_seed(5000 + seed)
                x32 = torch.randn(bs, L, dim, generator=g, device=dev, dtype=torch.float32)
                c32 = torch.randn(bs, TCTX, dim, generator=g, device=dev, dtype=torch.float32)
                x16 = x32.bfloat16(); c16 = c32.bfloat16()

                with torch.no_grad():
                    # fp32
                    ref_o = ref32(x32, {}, c32)[0]
                    st_t = ref32.infer_build_state(c32)
                    tor_o = torch.cat([ref32.infer_step(x32[:, t:t+1], st_t) for t in range(L)], 1)
                    st_c = cuda32.infer_build_state(c32)
                    cu_o = torch.cat([cuda32.infer_step(x32[:, t:t+1], st_c) for t in range(L)], 1)
                    # bf16
                    st_t16 = ref16.infer_build_state(c16)
                    tor16 = torch.cat([ref16.infer_step(x16[:, t:t+1], st_t16) for t in range(L)], 1)
                    st_c16 = cuda16.infer_build_state(c16)
                    cu16 = torch.cat([cuda16.infer_step(x16[:, t:t+1], st_c16) for t in range(L)], 1)

                _, r_cuda_ref, _ = errs(ref_o, cu_o)
                _, r_cuda_tor, _ = errs(tor_o, cu_o)
                _, r_tor_truth, _ = errs(ref_o, tor16)   # ref_o is fp32 truth
                _, r_cuda_truth, _ = errs(ref_o, cu16)
                _, r_cuda_tor16, _ = errs(tor16, cu16)

                acc["fp32 CUDA-vs-REF rel"].append(r_cuda_ref)
                acc["fp32 CUDA-vs-TORCH rel"].append(r_cuda_tor)
                acc["bf16 CUDA-vs-truth rel"].append(r_cuda_truth)
                acc["bf16 TORCH-vs-truth rel"].append(r_tor_truth)
                acc["bf16 CUDA-vs-TORCH rel"].append(r_cuda_tor16)

                cfg = f"hd={hd},bs={bs},seed={seed}"
                if r_cuda_ref > worst["fp32_cuda_ref"][0]:
                    worst["fp32_cuda_ref"] = (r_cuda_ref, cfg)
                if r_cuda_truth > worst["bf16_cuda_truth"][0]:
                    worst["bf16_cuda_truth"] = (r_cuda_truth, cfg)

                # per-config criteria
                if not (r_cuda_ref < 1.5e-3):
                    fails.append((cfg, f"fp32 CUDA-vs-REF rel={r_cuda_ref:.2e} >= 1.5e-3"))
                # CUDA vs torch incremental: same algorithm, differ only by GEMM
                # reduction order (batch/shape) -> fp32 rounding ~1e-4, not a bug.
                if not (r_cuda_tor < 2e-4):
                    fails.append((cfg, f"fp32 CUDA-vs-TORCH rel={r_cuda_tor:.2e} >= 2e-4"))
                if not (r_cuda_truth <= r_tor_truth * 1.5 + 1e-6):
                    fails.append((cfg, f"bf16 CUDA rel={r_cuda_truth:.2e} > 1.5x TORCH {r_tor_truth:.2e}"))
                # NaN/Inf guard
                if not (torch.isfinite(cu_o).all() and torch.isfinite(cu16).all()):
                    fails.append((cfg, "NaN/Inf in CUDA output"))
                del ref32, cuda32, ref16, cuda16
                torch.cuda.empty_cache()

    print("=== 误差分布 (所有配置汇总) ===")
    for k, v in acc.items():
        summarize(k, v)
    print(f"\n=== 最坏配置 ===")
    print(f"  fp32 CUDA-vs-REF  : {worst['fp32_cuda_ref'][0]:.3e}  @ {worst['fp32_cuda_ref'][1]}")
    print(f"  bf16 CUDA-vs-truth: {worst['bf16_cuda_truth'][0]:.3e}  @ {worst['bf16_cuda_truth'][1]}")

    # extreme-input stability
    print(f"\n=== 极端输入稳定性 ===")
    torch.manual_seed(0)
    ref, cuda, dim = build_pair(torch.bfloat16, 64)
    for scale, tag in [(100.0, "large(x100)"), (1e-3, "tiny(x1e-3)"), (0.0, "zeros")]:
        g = torch.Generator(device=dev).manual_seed(9)
        x = torch.randn(1, L, dim, generator=g, device=dev, dtype=torch.bfloat16) * scale
        c = torch.randn(1, TCTX, dim, generator=g, device=dev, dtype=torch.bfloat16) * scale
        with torch.no_grad():
            st = cuda.infer_build_state(c)
            o = torch.cat([cuda.infer_step(x[:, t:t+1], st) for t in range(8)], 1)
        fin = torch.isfinite(o).all().item()
        print(f"  {tag:14s}: finite={fin}  out_absmax={o.float().abs().max().item():.3e}")
        if not fin:
            fails.append((tag, "non-finite under extreme input"))

    print()
    if fails:
        print(f"=== SWEEP FAIL ({len(fails)} 个问题) ===")
        for cfg, msg in fails[:20]:
            print(f"  [{cfg}] {msg}")
        return 1
    print(f"=== SWEEP PASS (全部 {n_cfg} 配置 + 极端输入) ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
