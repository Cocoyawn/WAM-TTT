"""
大规模 ~1 小时验证: 持续累积 TTT-256 推理 CUDA kernel 的正确性误差 + 三层速度,
增量写 JSON (防中断). 跑满 DURATION_S 后停止.

正确性: 每个 (seed, batch, head_dim) 配置记录:
  fp32  CUDA-vs-REF rel/abs
  bf16  CUDA-vs-fp32truth rel,  TORCH-vs-fp32truth rel
速度: 周期性测 attention / torch-TTT-256 / CUDA-TTT-256 两场景 (单次 forward + 256步 rollout),
  CUDA events, 中位数+std.

输出 JSON: docs/ttt_validation_data.json  (供 plot_ttt_validation.py 出图出表)

Run on a GPU (1h):
  TORCHDYNAMO_DISABLE=1 PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES=<x> \
    /mnt/afs-h200/yuyangcheng/venvs/fla_triton32/bin/python -m scripts.validate_ttt_1h
"""
import json
import os
import time
import statistics as st
import torch
from src.models.ttt import FastWeightGluMLPMultihead

DEV = "cuda"
DURATION_S = float(os.environ.get("VALIDATE_SECONDS", "3600"))  # 1 hour
HEADS, L, TCTX = 12, 256, 16
BATCHES = [1, 2, 4]
HEAD_DIMS = [32, 64, 128]
OUT = "docs/ttt_validation_data.json"
SPEED_EVERY_S = 120.0  # run speed bench every ~2 min

# fixed elapsed clock from a base (Date.now unavailable inside; use time.time here is fine - this is a normal script)
_T0 = time.time()


def errs(ref, x):
    ref = ref.float(); x = x.float()
    d = (ref - x).abs()
    amax = d.max().item()
    rmax = amax / (ref.abs().max().item() + 1e-12)
    return amax, rmax


def build_pair(dtype, head_dim, seed):
    dim = head_dim * HEADS
    torch.manual_seed(seed)
    ref = FastWeightGluMLPMultihead(dim=dim, head_dim=head_dim, causal=True, chunk_size=256,
                                    vlm_hidden_size=dim, use_cuda_kernel=False).to(DEV, dtype).eval()
    torch.manual_seed(seed)
    cuda = FastWeightGluMLPMultihead(dim=dim, head_dim=head_dim, causal=True, chunk_size=256,
                                     vlm_hidden_size=dim, use_cuda_kernel=True).to(DEV, dtype).eval()
    cuda.load_state_dict(ref.state_dict())
    return ref, cuda, dim


@torch.no_grad()
def one_config(seed, bs, hd):
    ref32, cuda32, dim = build_pair(torch.float32, hd, 1000 + seed)
    ref16, cuda16, _ = build_pair(torch.bfloat16, hd, 1000 + seed)
    g = torch.Generator(device=DEV).manual_seed(5000 + seed)
    x32 = torch.randn(bs, L, dim, generator=g, device=DEV, dtype=torch.float32)
    c32 = torch.randn(bs, TCTX, dim, generator=g, device=DEV, dtype=torch.float32)
    x16, c16 = x32.bfloat16(), c32.bfloat16()
    ref_o = ref32(x32, {}, c32)[0]                       # fp32 truth
    st_c = cuda32.infer_build_state(c32)
    cu_o = torch.cat([cuda32.infer_step(x32[:, t:t+1], st_c) for t in range(L)], 1)
    st_t16 = ref16.infer_build_state(c16)
    tor16 = torch.cat([ref16.infer_step(x16[:, t:t+1], st_t16) for t in range(L)], 1)
    st_c16 = cuda16.infer_build_state(c16)
    cu16 = torch.cat([cuda16.infer_step(x16[:, t:t+1], st_c16) for t in range(L)], 1)
    a_fp32, r_fp32 = errs(ref_o, cu_o)
    _, r_bf16_cuda = errs(ref_o, cu16)
    _, r_bf16_torch = errs(ref_o, tor16)
    finite = bool(torch.isfinite(cu_o).all() and torch.isfinite(cu16).all())
    del ref32, cuda32, ref16, cuda16
    torch.cuda.empty_cache()
    return dict(seed=seed, bs=bs, hd=hd, abs_fp32=a_fp32, rel_fp32=r_fp32,
                rel_bf16_cuda=r_bf16_cuda, rel_bf16_torch=r_bf16_torch, finite=finite)


def cuda_time(fn, reps, warmup):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return ts


@torch.no_grad()
def speed_sample():
    torch.manual_seed(0)
    dim = HEADS * 64
    attn = torch.nn.MultiheadAttention(dim, HEADS, batch_first=True).to(DEV, torch.bfloat16).eval()
    ttt_t = FastWeightGluMLPMultihead(dim=dim, head_dim=64, causal=True, chunk_size=256,
                                      vlm_hidden_size=dim, use_cuda_kernel=False).to(DEV, torch.bfloat16).eval()
    ttt_c = FastWeightGluMLPMultihead(dim=dim, head_dim=64, causal=True, chunk_size=256,
                                      vlm_hidden_size=dim, use_cuda_kernel=True).to(DEV, torch.bfloat16).eval()
    g = torch.Generator(device=DEV).manual_seed(1)
    x = torch.randn(1, L, dim, generator=g, device=DEV, dtype=torch.bfloat16)
    ctx = torch.randn(1, TCTX, dim, generator=g, device=DEV, dtype=torch.bfloat16)
    vfeat = torch.randn(1, TCTX, dim, generator=g, device=DEV, dtype=torch.bfloat16)

    def amask(t):
        m = torch.zeros((t, t + TCTX), device=DEV, dtype=torch.bfloat16)
        cm = torch.triu(torch.ones((t, t), device=DEV, dtype=torch.bool), 1)
        m[:, :t].masked_fill_(cm, float('-inf')); return m
    def attn_full():
        kv = torch.cat([x, vfeat], 1); attn(x, kv, kv, attn_mask=amask(L))
    def attn_roll():
        for t in range(1, L + 1):
            xt = x[:, :t]; kv = torch.cat([xt, vfeat], 1); attn(xt, kv, kv, attn_mask=amask(t))
    def torch_full():
        ttt_t(x, {}, ctx)
    def torch_roll():
        for t in range(1, L + 1): ttt_t(x[:, :t], {}, ctx)
    def cuda_full():
        s = ttt_c.infer_build_state(ctx); ttt_c.infer_step(x, s)
    def cuda_roll():
        s = ttt_c.infer_build_state(ctx)
        for t in range(L): ttt_c.infer_step(x[:, t:t+1], s)

    out = {"A_full": {}, "B_rollout": {}}
    for name, fn in [("attention", attn_full), ("torch_ttt", torch_full), ("cuda_ttt", cuda_full)]:
        ts = cuda_time(fn, reps=30, warmup=5)
        out["A_full"][name] = dict(median=st.median(ts), mean=sum(ts)/len(ts), std=st.pstdev(ts), min=min(ts))
    for name, fn in [("attention", attn_roll), ("torch_ttt", torch_roll), ("cuda_ttt", cuda_roll)]:
        ts = cuda_time(fn, reps=4, warmup=1)
        out["B_rollout"][name] = dict(median=st.median(ts), mean=sum(ts)/len(ts), std=st.pstdev(ts), min=min(ts))
    del attn, ttt_t, ttt_c; torch.cuda.empty_cache()
    return out


def gpu_used_mb():
    free, total = torch.cuda.mem_get_info()
    return (total - free) / 1e6


def save(data):
    os.makedirs("docs", exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(data, f)


def main():
    data = {"meta": {"duration_s": DURATION_S, "L": L, "heads": HEADS,
                     "batches": BATCHES, "head_dims": HEAD_DIMS,
                     "gpu": torch.cuda.get_device_name(0),
                     "gpu_used_mb_start": gpu_used_mb()},
            "correctness": [], "speed_samples": []}
    seed = 0
    last_speed = -1e9  # force first speed sample after the first correctness batch
    while time.time() - _T0 < DURATION_S:
        for bs in BATCHES:
            for hd in HEAD_DIMS:
                rec = one_config(seed, bs, hd)
                rec["t"] = time.time() - _T0
                data["correctness"].append(rec)
        seed += 1
        now = time.time() - _T0
        if now - last_speed >= SPEED_EVERY_S:
            data["speed_samples"].append({"t": now, **speed_sample()})
            last_speed = now
        save(data)  # incremental
        n = len(data["correctness"])
        if seed % 5 == 0:
            print(f"[{now:7.1f}s] seeds={seed} configs={n} speed_samples={len(data['speed_samples'])} "
                  f"gpu_used={gpu_used_mb():.0f}MiB", flush=True)
    data["meta"]["gpu_used_mb_end"] = gpu_used_mb()
    data["meta"]["total_configs"] = len(data["correctness"])
    data["meta"]["total_seeds"] = seed
    save(data)
    print(f"DONE: {len(data['correctness'])} configs, {len(data['speed_samples'])} speed samples "
          f"over {time.time()-_T0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
