"""
Per-layer TTT kernel profiler: isolate ONE causal vision-expert TTT layer at the
real shapes and profile forward+backward kernel time, torch-op vs CUDA-kernel.

Shows where the time goes (the kernel-launch storm of the eager torch path vs
the fused CUDA path), independent of the full VLANeXt rollout.

Run (needs a CLEAN, EXCLUSIVE GPU for meaningful timings):
  TORCHDYNAMO_DISABLE=1 PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES=<clean gpu> \
    /mnt/afs-h200/yuyangcheng/venvs/fla_triton32/bin/python -m scripts.bench_ttt_layer_profile
"""
import os
import time
import torch

from src.models.ttt import FastWeightGluMLPMultihead

DEV, DT = "cuda", torch.bfloat16
# real vision-expert shapes
HIDDEN, HEADS = 768, 12          # head_dim=64
B, L, T_CTX = 8, 256, 16         # batch, image tokens, vlm ctx tokens
CHUNK = 256
WARMUP, ITERS = 10, 50


def make_layer(use_cuda):
    torch.manual_seed(0)
    layer = FastWeightGluMLPMultihead(
        dim=HIDDEN, head_dim=HIDDEN // HEADS, causal=True, chunk_size=CHUNK,
        vlm_hidden_size=HIDDEN, use_cuda_kernel=use_cuda,
    ).to(DEV, DT)
    return layer


def make_inputs():
    g = torch.Generator(device=DEV).manual_seed(1)
    x = torch.randn(B, L, HIDDEN, generator=g, device=DEV, dtype=DT, requires_grad=True)
    ctx = torch.randn(B, T_CTX, HIDDEN, generator=g, device=DEV, dtype=DT, requires_grad=True)
    return x, ctx


def timed_fwd_bwd(layer, x, ctx):
    def step():
        out, _ = layer(x, {}, ctx)
        loss = out.float().pow(2).mean()
        g, = torch.autograd.grad(loss, x, retain_graph=False)
        return g
    for _ in range(WARMUP):
        step()
    torch.cuda.synchronize(); t = time.time()
    for _ in range(ITERS):
        step()
    torch.cuda.synchronize()
    return (time.time() - t) / ITERS * 1000  # ms


def profile_one(layer, x, ctx, tag):
    from torch.profiler import profile, ProfilerActivity
    def step():
        out, _ = layer(x, {}, ctx)
        loss = out.float().pow(2).mean()
        torch.autograd.grad(loss, x)
    for _ in range(5):
        step()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as prof:
        for _ in range(10):
            step()
        torch.cuda.synchronize()
    print(f"\n===== {tag}: top CUDA kernels (fwd+bwd, 10 iters) =====")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=12))


if __name__ == "__main__":
    compile_off = os.environ.get("TORCHDYNAMO_DISABLE") == "1"
    print(f"single causal TTT layer | dim={HIDDEN} heads={HEADS} B={B} L={L} "
          f"chunk={CHUNK} | compile={'OFF' if compile_off else 'ON'} | dtype=bf16\n")
    x, ctx = make_inputs()
    for use_cuda, label in [(False, "torch-TTT"), (True, "CUDA-TTT ")]:
        layer = make_layer(use_cuda)
        ms = timed_fwd_bwd(layer, x, ctx)
        print(f"[{label}] fwd+bwd {ms:8.3f} ms/iter")
        del layer; torch.cuda.empty_cache()

    # kernel-level breakdown
    for use_cuda, label in [(False, "torch-TTT"), (True, "CUDA-TTT")]:
        layer = make_layer(use_cuda)
        profile_one(layer, x, ctx, label)
        del layer; torch.cuda.empty_cache()
