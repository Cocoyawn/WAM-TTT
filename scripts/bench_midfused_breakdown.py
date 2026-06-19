"""
Stage-2 决策用: 当前 mid-fused infer_step 的逻辑步骤时间分布.
mid-kernel 已把 q-norm+apply+RMSNorm 融成 1 个. 现在 infer_step 只剩 3 步:
  to_qkv (q-only Linear) -> mid-kernel(融合) -> c_proj (Linear)
看现在谁是瓶颈, 决定要不要把两个 Linear 也融进去.

也单独基准: 手写 768-matvec (单 block) vs cuBLAS F.linear, 看融合 Linear 划不划算.

Run on CLEAN GPU:
  TORCHDYNAMO_DISABLE=1 PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES=<clean> \
    /mnt/afs-h200/yuyangcheng/venvs/fla_triton32/bin/python -m scripts.bench_midfused_breakdown
"""
import statistics as st
import torch
import torch.nn.functional as F
from einops import rearrange
from src.models.ttt import FastWeightGluMLPMultihead

DEV, DT = "cuda", torch.bfloat16
HIDDEN, HEADS, TCTX = 768, 12, 16
HD = HIDDEN // HEADS
REPS, WARMUP = 300, 40


def ct(fn, reps=REPS):
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) * 1000)
    return st.median(ts)


@torch.no_grad()
def main():
    free, total = torch.cuda.mem_get_info()
    used = (total - free) / 1e6
    print(f"mid-fused infer_step breakdown | bf16 | GPU~{used:.0f}MiB "
          + ("[clean]" if used < 2000 else "[WARN]"))
    print(f"({REPS} reps median, us)\n")

    torch.manual_seed(0)
    layer = FastWeightGluMLPMultihead(dim=HIDDEN, head_dim=HD, causal=True, chunk_size=256,
                                      vlm_hidden_size=HIDDEN, use_cuda_kernel=True).to(DEV, DT).eval()
    from src.models.ttt_cuda import _load_extension
    ext = _load_extension()
    g = torch.Generator(device=DEV).manual_seed(1)
    x = torch.randn(1, 1, HIDDEN, generator=g, device=DEV, dtype=DT)
    ctx = torch.randn(1, TCTX, HIDDEN, generator=g, device=DEV, dtype=DT)
    state = layer.infer_build_state(ctx)
    w0, w1, w2 = state["w0"], state["w1"], state["w2"]
    Wq = layer.to_qkv.weight[:HIDDEN]
    bq = layer.to_qkv.bias[:HIDDEN] if layer.to_qkv.bias is not None else None
    onw = layer.o_norm.weight.to(DT)

    def s_toqkv():
        return F.silu(F.linear(x, Wq, bq), inplace=False)
    q_lin = s_toqkv()
    q_un = rearrange(q_lin, "b l (h d) -> (b h) (l d)", h=HEADS)
    def s_mid():
        return ext.infer_step_mid(q_un, w0, w2, w1, onw, 1e-5, 1e-5)
    o = s_mid()
    o_r = rearrange(o.unsqueeze(1), "(b h) l d -> b l (h d)", h=HEADS, b=1)
    def s_cproj():
        return layer.c_proj(o_r)

    t_qkv = ct(s_toqkv); t_mid = ct(s_mid); t_cp = ct(s_cproj)
    t_e2e = ct(lambda: layer.infer_step(x, state))
    parts = [("to_qkv (q-only Linear)", t_qkv), ("mid-kernel (fused)", t_mid),
             ("c_proj (Linear)", t_cp)]
    s = sum(p[1] for p in parts)
    print(f"{'step':28s} {'us':>8s} {'%':>6s}")
    for nm, us in parts:
        print(f"  {nm:26s} {us:8.2f} {100*us/s:6.1f}")
    print(f"  {'-'*26} {'-'*8}")
    print(f"  {'sum of parts':26s} {s:8.2f}")
    print(f"  {'end-to-end':26s} {t_e2e:8.2f}   (gap={t_e2e-s:+.2f}us)")

    # matvec micro-bench: cuBLAS vs naive, for the Stage-2 decision
    print("\n--- 768x768 matvec: cuBLAS F.linear vs (reference for fusion) ---")
    W = torch.randn(HIDDEN, HIDDEN, device=DEV, dtype=DT)
    v = torch.randn(1, HIDDEN, device=DEV, dtype=DT)
    t_cublas = ct(lambda: F.linear(v, W))
    print(f"  cuBLAS F.linear(1x768 @ 768x768): {t_cublas:.2f} us")
    print(f"  => Linear 两步合计 {t_qkv+t_cp:.1f}us; 若融合能省 launch+间隙, 潜在收益 ~{(t_qkv+t_cp)*0.5:.0f}us")


if __name__ == "__main__":
    main()
