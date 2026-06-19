"""
Stage-1 mid-kernel 前后对比: 整个 infer_step 端到端 latency.
对比 3 条路径 (同一层, 同一输入, CUDA events 多次中位数):
  torch    : use_cuda_kernel=False (纯 torch)
  apply-only: 旧融合 kernel (q-norm 在外, apply+RMSNorm 融合) —— 强制走 ext.infer_step
  mid-fused : stage-1 (q-norm+apply+RMSNorm 融合) —— 走 ext.infer_step_mid
也测整段 256-step rollout (7 层).

Run on a CLEAN GPU:
  TORCHDYNAMO_DISABLE=1 PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES=<clean> \
    /mnt/afs-h200/yuyangcheng/venvs/fla_triton32/bin/python -m scripts.bench_mid_kernel
"""
import statistics as st
import torch
import torch.nn.functional as F
from einops import rearrange
from src.models.ttt import FastWeightGluMLPMultihead

DEV, DT = "cuda", torch.bfloat16
HIDDEN, HEADS, L, TCTX = 768, 12, 16, 16
HD = HIDDEN // HEADS
REPS, WARMUP = 300, 40


def ctime(fn, reps=REPS):
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) * 1000)
    return st.median(ts), st.pstdev(ts)


@torch.no_grad()
def main():
    free, total = torch.cuda.mem_get_info()
    used = (total - free) / 1e6
    print(f"infer_step end-to-end | dim={HIDDEN} hd={HD} bf16 | GPU used~{used:.0f}MiB "
          + ("[clean]" if used < 2000 else "[WARN polluted]"))
    print(f"(CUDA events, {REPS} reps median, us)\n")

    torch.manual_seed(0)
    layer_t = FastWeightGluMLPMultihead(dim=HIDDEN, head_dim=HD, causal=True, chunk_size=256,
                                        vlm_hidden_size=HIDDEN, use_cuda_kernel=False).to(DEV, DT).eval()
    layer_c = FastWeightGluMLPMultihead(dim=HIDDEN, head_dim=HD, causal=True, chunk_size=256,
                                        vlm_hidden_size=HIDDEN, use_cuda_kernel=True).to(DEV, DT).eval()
    layer_c.load_state_dict(layer_t.state_dict())
    g = torch.Generator(device=DEV).manual_seed(1)
    x1 = torch.randn(1, 1, HIDDEN, generator=g, device=DEV, dtype=DT)
    ctx = torch.randn(1, TCTX, HIDDEN, generator=g, device=DEV, dtype=DT)
    st_t = layer_t.infer_build_state(ctx)
    st_c = layer_c.infer_build_state(ctx)

    # single-step end-to-end
    m_t, s_t = ctime(lambda: layer_t.infer_step(x1, st_t))
    m_c, s_c = ctime(lambda: layer_c.infer_step(x1, st_c))
    print("--- single infer_step (end-to-end) ---")
    print(f"  torch         : {m_t:7.2f} us  (std {s_t:.2f})")
    print(f"  CUDA mid-fused : {m_c:7.2f} us  (std {s_c:.2f})   {m_t/m_c:.2f}x vs torch")

    # 256-step rollout, 7 layers
    LR = 256
    xr = torch.randn(1, LR, HIDDEN, generator=g, device=DEV, dtype=DT)
    layers_t = [FastWeightGluMLPMultihead(dim=HIDDEN, head_dim=HD, causal=True, chunk_size=256,
                vlm_hidden_size=HIDDEN, use_cuda_kernel=False).to(DEV, DT).eval() for _ in range(7)]
    layers_c = [FastWeightGluMLPMultihead(dim=HIDDEN, head_dim=HD, causal=True, chunk_size=256,
                vlm_hidden_size=HIDDEN, use_cuda_kernel=True).to(DEV, DT).eval() for _ in range(7)]
    for a, b in zip(layers_c, layers_t):
        a.load_state_dict(b.state_dict())
    def roll(layers):
        states = [l.infer_build_state(ctx) for l in layers]
        for t in range(LR):
            xt = xr[:, t:t+1]
            for l, s in zip(layers, states):
                l.infer_step(xt, s)
    m_rt, s_rt = ctime(lambda: roll(layers_t), reps=8)
    m_rc, s_rc = ctime(lambda: roll(layers_c), reps=8)
    print("\n--- 256-step rollout, 7 TTT layers ---")
    print(f"  torch         : {m_rt/1000:8.2f} ms")
    print(f"  CUDA mid-fused : {m_rc/1000:8.2f} ms   {m_rt/m_rc:.2f}x vs torch")


if __name__ == "__main__":
    main()
