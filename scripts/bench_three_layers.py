"""
严谨单层推理速度对比: attention vs torch-TTT-256 vs CUDA-TTT-256.

测两个场景, 每个用 CUDA events 计时, warmup + 多次重复, 报告 中位数/均值/std/min:
  A) 整段前缀一次 forward (L=256):  layer(x[:256])  -- 一次性处理全序列
  B) 自回归 256 步 rollout 累计:  逐步处理 (attn/torch-TTT 重算前缀; CUDA-TTT 增量)
     -> 这才是部署真实代价。

attention 和 torch-TTT 走普通 forward(无 cache, 每步重算前缀);
CUDA-TTT 走增量(build_state 一次 + infer_step 每步)。三者数学等价(已对拍)。

Run on a CLEAN GPU (绝对计时需独占卡):
  TORCHDYNAMO_DISABLE=1 PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES=<clean> \
    /mnt/afs-h200/yuyangcheng/venvs/fla_triton32/bin/python -m scripts.bench_three_layers
"""
import statistics as st
import torch
from src.models.ttt import FastWeightGluMLPMultihead

DEV, DT = "cuda", torch.bfloat16
HIDDEN, HEADS, L, TCTX = 768, 12, 256, 16
REPS = 30
WARMUP = 5


def cuda_time(fn, reps=REPS, warmup=WARMUP):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))  # ms
    return ts


def report(tag, ts, ref_med=None):
    med = st.median(ts); mean = sum(ts)/len(ts); sd = st.pstdev(ts); mn = min(ts)
    sp = f"  {ref_med/med:5.2f}x" if ref_med else ""
    print(f"  {tag:22s} median={med:9.3f} mean={mean:9.3f} std={sd:7.3f} min={mn:9.3f} ms{sp}")
    return med


def make(mixer_cuda):
    torch.manual_seed(0)
    if mixer_cuda == "attn":
        return torch.nn.MultiheadAttention(HIDDEN, HEADS, batch_first=True).to(DEV, DT).eval()
    return FastWeightGluMLPMultihead(
        dim=HIDDEN, head_dim=HIDDEN // HEADS, causal=True, chunk_size=256,
        vlm_hidden_size=HIDDEN, use_cuda_kernel=(mixer_cuda == "cuda")).to(DEV, DT).eval()


@torch.no_grad()
def main():
    free, total = torch.cuda.mem_get_info()
    used = (total - free) / 1e6
    print(f"single layer inference | dim={HIDDEN} heads={HEADS} L={L} bf16 | "
          f"GPU used~{used:.0f}MiB" + ("  [WARN polluted]" if used > 2000 else "  [clean]"))
    print(f"(CUDA events, {REPS} reps, {WARMUP} warmup)\n")

    g = torch.Generator(device=DEV).manual_seed(1)
    x = torch.randn(1, L, HIDDEN, generator=g, device=DEV, dtype=DT)
    ctx = torch.randn(1, TCTX, HIDDEN, generator=g, device=DEV, dtype=DT)
    attn = make("attn")
    ttt_t = make("torch")
    ttt_c = make("cuda")
    # attention needs kv = cat([x, vlm]) with causal mask; build once-style per call
    vfeat = torch.randn(1, TCTX, HIDDEN, generator=g, device=DEV, dtype=DT)

    def attn_full():
        T = x.shape[1]
        mask = torch.zeros((T, T + TCTX), device=DEV, dtype=DT)
        cm = torch.triu(torch.ones((T, T), device=DEV, dtype=torch.bool), 1)
        mask[:, :T].masked_fill_(cm, float('-inf'))
        kv = torch.cat([x, vfeat], 1)
        attn(x, kv, kv, attn_mask=mask)

    print("--- 场景 A: 整段前缀一次 forward (L=256) ---")
    m_attn = report("attention", cuda_time(attn_full))
    report("torch-TTT-256", cuda_time(lambda: ttt_t(x, {}, ctx)), m_attn)
    # CUDA-TTT one-shot == batch infer_step (build state once + apply all)
    def cuda_full():
        s = ttt_c.infer_build_state(ctx); ttt_c.infer_step(x, s)
    report("CUDA-TTT-256", cuda_time(cuda_full), m_attn)

    print("\n--- 场景 B: 自回归 256 步 rollout 累计 (部署真实代价) ---")
    def attn_roll():
        for t in range(1, L + 1):
            xt = x[:, :t]
            mask = torch.zeros((t, t + TCTX), device=DEV, dtype=DT)
            cm = torch.triu(torch.ones((t, t), device=DEV, dtype=torch.bool), 1)
            mask[:, :t].masked_fill_(cm, float('-inf'))
            kv = torch.cat([xt, vfeat], 1)
            attn(xt, kv, kv, attn_mask=mask)
    def torch_roll():
        for t in range(1, L + 1):
            ttt_t(x[:, :t], {}, ctx)
    def cuda_roll():
        s = ttt_c.infer_build_state(ctx)
        for t in range(L):
            ttt_c.infer_step(x[:, t:t+1], s)
    m_attn_r = report("attention", cuda_time(attn_roll, reps=10, warmup=2))
    report("torch-TTT-256", cuda_time(torch_roll, reps=10, warmup=2), m_attn_r)
    report("CUDA-TTT-256", cuda_time(cuda_roll, reps=10, warmup=2), m_attn_r)


if __name__ == "__main__":
    main()
