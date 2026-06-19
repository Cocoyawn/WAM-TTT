"""
逻辑步骤级 profile: infer_step 单步的时间到底花在哪一步。
拆成: to_qkv(q-only Linear) / q-norm / fused-apply(CUDA kernel) / c_proj.
每段单独计时 (CUDA events, 多次重复中位数), 在干净卡上跑才可信.

也对比: 整个 infer_step 端到端 vs 各段之和 (看 launch 间隙开销).

Run on a CLEAN GPU:
  TORCHDYNAMO_DISABLE=1 PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES=<clean> \
    /mnt/afs-h200/yuyangcheng/venvs/fla_triton32/bin/python -m scripts.bench_infer_step_breakdown
"""
import statistics as st
import torch
import torch.nn.functional as F
from einops import rearrange
from src.models.ttt import FastWeightGluMLPMultihead

DEV, DT = "cuda", torch.bfloat16
HIDDEN, HEADS, TCTX = 768, 12, 16
HD = HIDDEN // HEADS
REPS, WARMUP = 200, 30


def ctime(fn):
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(REPS):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) * 1000)  # us
    return st.median(ts)


@torch.no_grad()
def main():
    free, total = torch.cuda.mem_get_info()
    used = (total - free) / 1e6
    print(f"infer_step breakdown | dim={HIDDEN} heads={HEADS} hd={HD} bf16 | "
          f"GPU used~{used:.0f}MiB " + ("[WARN polluted]" if used > 2000 else "[clean]"))
    print(f"(CUDA events, {REPS} reps median, us)\n")

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

    # --- 分段 ---
    def s_to_qkv():
        return F.silu(F.linear(x, Wq, bq), inplace=False)
    q_lin = s_to_qkv()
    def s_qnorm():
        q = rearrange(q_lin, "b l (h d) -> (b h) l d", h=HEADS)
        return q / (q.norm(dim=2, keepdim=True) + 1e-5).to(DT)
    qn = s_qnorm().squeeze(1).contiguous()
    def s_apply():
        return ext.infer_step(qn, w0, w2, w1, onw, 1e-5)
    o = s_apply()
    o_r = o.unsqueeze(1)
    def s_cproj():
        out = rearrange(o_r, "(b h) l d -> b l (h d)", h=HEADS, b=1)
        return layer.c_proj(out)

    t_qkv = ctime(s_to_qkv)
    t_norm = ctime(s_qnorm)
    t_apply = ctime(s_apply)
    t_cproj = ctime(s_cproj)
    t_end2end = ctime(lambda: layer.infer_step(x, state))

    parts = [("to_qkv (q-only Linear)", t_qkv), ("q-norm", t_norm),
             ("fused-apply (CUDA kernel)", t_apply), ("c_proj (Linear)", t_cproj)]
    s = sum(p[1] for p in parts)
    print(f"{'step':28s} {'us':>8s} {'%':>6s}")
    for name, us in parts:
        print(f"  {name:26s} {us:8.2f} {100*us/s:6.1f}")
    print(f"  {'-'*26} {'-'*8}")
    print(f"  {'sum of parts':26s} {s:8.2f}")
    print(f"  {'end-to-end infer_step':26s} {t_end2end:8.2f}   (gap={t_end2end-s:+.2f}us = python/launch 间隙)")
    print(f"\n判断: 占比最大的那步是真瓶颈; 若 'fused-apply' 已经很小,")
    print(f"      则继续优化 kernel 没用, 大头在 Linear + python 间隙 + launch.")


if __name__ == "__main__":
    main()
