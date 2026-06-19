"""
Inference latency: full-recompute vs incremental TTT layers over a 256-step
autoregressive rollout. Isolates the 7 vision-expert TTT layers (the inference
bottleneck per test_infer_claims) to show the O(n^2)->O(n) win.

full-recompute: each step t re-runs the TTT layer on the length-t prefix (what
the current predict_action loop does).
incremental: build_state(ctx) ONCE, then infer_step on the single new token each
step (O(1)/step), valid because chunk=256 makes positions independent.

Run on a CLEAN GPU:
  TORCHDYNAMO_DISABLE=1 PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES=<clean> \
    /mnt/afs-h200/yuyangcheng/venvs/fla_triton32/bin/python -m scripts.bench_ttt_incremental_infer
"""
import time
import torch
from src.models.ttt import FastWeightGluMLPMultihead

DEV, DT = "cuda", torch.bfloat16
HIDDEN, HEADS, L, TCTX = 768, 12, 256, 16
N_TTT = 7  # vision-expert TTT layers
WARMUP, ITERS = 2, 5


def make_layers(use_cuda=False):
    torch.manual_seed(0)
    return [FastWeightGluMLPMultihead(
        dim=HIDDEN, head_dim=HIDDEN // HEADS, causal=True, chunk_size=256,
        vlm_hidden_size=HIDDEN, use_cuda_kernel=use_cuda).to(DEV, DT).eval()
        for _ in range(N_TTT)]


@torch.no_grad()
def run_full(layers, x_full, ctx):
    # emulate AR: at step t, run each TTT layer on prefix [0..t]
    for t in range(1, L + 1):
        xt = x_full[:, :t]
        for lyr in layers:
            lyr(xt, {}, ctx)


@torch.no_grad()
def run_incremental(layers, x_full, ctx):
    states = [lyr.infer_build_state(ctx) for lyr in layers]  # once
    for t in range(L):
        xt = x_full[:, t:t + 1]
        for lyr, st in zip(layers, states):
            lyr.infer_step(xt, st)


@torch.no_grad()
def timed(fn, *a):
    for _ in range(WARMUP):
        fn(*a)
    torch.cuda.synchronize(); t = time.time()
    for _ in range(ITERS):
        fn(*a)
    torch.cuda.synchronize()
    return (time.time() - t) / ITERS * 1000


if __name__ == "__main__":
    free, total = torch.cuda.mem_get_info()
    used_mb = (total - free) / 1e6
    print(f"7 vision-TTT layers | 256-step AR rollout | bf16 | GPU used≈{used_mb:.0f}MiB"
          + ("  [WARN polluted >2GB]" if used_mb > 2000 else ""))
    layers = make_layers()
    g = torch.Generator(device=DEV).manual_seed(1)
    x_full = torch.randn(1, L, HIDDEN, generator=g, device=DEV, dtype=DT)
    ctx = torch.randn(1, TCTX, HIDDEN, generator=g, device=DEV, dtype=DT)

    t_full = timed(run_full, layers, x_full, ctx)
    t_inc = timed(run_incremental, layers, x_full, ctx)
    layers_cuda = make_layers(use_cuda=True)
    t_inc_cuda = timed(run_incremental, layers_cuda, x_full, ctx)
    print(f"  full-recompute (O(n^2))      : {t_full:9.1f} ms/rollout  (torch-TTT-256 baseline)")
    print(f"  incremental    torch  (O(n)) : {t_inc:9.1f} ms/rollout  {t_full/t_inc:5.1f}x vs full")
    print(f"  incremental    CUDA   (O(n)) : {t_inc_cuda:9.1f} ms/rollout  {t_full/t_inc_cuda:5.1f}x vs full"
          f"  | {t_inc/t_inc_cuda:4.2f}x vs torch-incremental")
