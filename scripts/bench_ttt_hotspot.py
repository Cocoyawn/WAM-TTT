"""
Focused TTT-layer hotspot profiler — fwd and bwd separately, torch vs CUDA.

Prints, for each (variant, phase): total CUDA time, total #kernel launches, and
the top-8 kernels by cuda_time. This is the profile-guided-optimization input:
shows whether CUDA-TTT loses to torch on GEMM time, elementwise, or launch count.

Run on a CLEAN GPU:
  TORCHDYNAMO_DISABLE=1 PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES=<clean> \
    /mnt/afs-h200/yuyangcheng/venvs/fla_triton32/bin/python -m scripts.bench_ttt_hotspot
"""
import os
import torch
from torch.profiler import profile, ProfilerActivity

from src.models.ttt import FastWeightGluMLPMultihead

DEV, DT = "cuda", torch.bfloat16
HIDDEN, HEADS = 768, 12
B, L, T_CTX = 8, 256, 16
CHUNK = 256
ITERS = 20


def make_layer(use_cuda):
    torch.manual_seed(0)
    return FastWeightGluMLPMultihead(
        dim=HIDDEN, head_dim=HIDDEN // HEADS, causal=True, chunk_size=CHUNK,
        vlm_hidden_size=HIDDEN, use_cuda_kernel=use_cuda,
    ).to(DEV, DT)


def make_inputs():
    g = torch.Generator(device=DEV).manual_seed(1)
    x = torch.randn(B, L, HIDDEN, generator=g, device=DEV, dtype=DT, requires_grad=True)
    ctx = torch.randn(B, T_CTX, HIDDEN, generator=g, device=DEV, dtype=DT, requires_grad=True)
    return x, ctx


def profile_phase(layer, x, ctx, want_bwd, tag):
    def fwd():
        out, _ = layer(x, {}, ctx)
        return out
    def step():
        out = fwd()
        if want_bwd:
            loss = out.float().pow(2).mean()
            torch.autograd.grad(loss, x, retain_graph=False)
    for _ in range(5):
        step()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(ITERS):
            step()
        torch.cuda.synchronize()
    evs = list(prof.key_averages())
    # torch renamed cuda_time_total -> device_time_total in newer versions
    def ctime(e):
        return getattr(e, "device_time_total", None) or getattr(e, "cuda_time_total", 0)
    evs = [e for e in evs if ctime(e) > 0]
    total_us = sum(ctime(e) for e in evs) / ITERS
    n_launch = sum(e.count for e in evs) / ITERS
    evs.sort(key=ctime, reverse=True)
    print(f"\n=== {tag} === total {total_us:8.1f} us/iter | {n_launch:6.1f} kernel-launches/iter")
    print(f"{'kernel':<48s} {'us/iter':>9s} {'n/iter':>7s}")
    for e in evs[:8]:
        nm = e.key[:46]
        print(f"{nm:<48s} {ctime(e)/ITERS:9.1f} {e.count/ITERS:7.1f}")


if __name__ == "__main__":
    print(f"single causal TTT layer | dim={HIDDEN} heads={HEADS} B={B} L={L} chunk={CHUNK} bf16")
    x, ctx = make_inputs()
    for use_cuda, label in [(False, "torch"), (True, "CUDA")]:
        layer = make_layer(use_cuda)
        profile_phase(layer, x, ctx, want_bwd=False, tag=f"{label}  FORWARD-only")
        profile_phase(layer, x, ctx, want_bwd=True,  tag=f"{label}  FWD+BWD")
        del layer; torch.cuda.empty_cache()
