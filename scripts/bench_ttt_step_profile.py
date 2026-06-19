"""
Profile the incremental TTT single-step (infer_step) to confirm it is
launch-bound and identify which ops to fuse into a CUDA kernel.

Single step shape: q is [B*heads, 1, head_dim] = [12, 1, 64] (B=1) -- tiny GEMVs.
Per step per layer: q@w0, q@w2, silu*, @w1, o_norm, c_proj, to_qkv. 256 steps x
7 layers = 1792 calls -> if launch-bound, a fused kernel wins big.

Run on a CLEAN GPU:
  TORCHDYNAMO_DISABLE=1 PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES=<clean> \
    /mnt/afs-h200/yuyangcheng/venvs/fla_triton32/bin/python -m scripts.bench_ttt_step_profile
"""
import torch
from torch.profiler import profile, ProfilerActivity
from src.models.ttt import FastWeightGluMLPMultihead

DEV, DT = "cuda", torch.bfloat16
HIDDEN, HEADS, TCTX = 768, 12, 16
STEPS = 256


def ctime(e):
    return getattr(e, "device_time_total", None) or getattr(e, "cuda_time_total", 0)


if __name__ == "__main__":
    torch.manual_seed(0)
    layer = FastWeightGluMLPMultihead(
        dim=HIDDEN, head_dim=HIDDEN // HEADS, causal=True, chunk_size=256,
        vlm_hidden_size=HIDDEN).to(DEV, DT).eval()
    g = torch.Generator(device=DEV).manual_seed(1)
    ctx = torch.randn(1, TCTX, HIDDEN, generator=g, device=DEV, dtype=DT)
    x1 = torch.randn(1, 1, HIDDEN, generator=g, device=DEV, dtype=DT)

    with torch.no_grad():
        state = layer.infer_build_state(ctx)
        for _ in range(20):
            layer.infer_step(x1, state)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(STEPS):
                layer.infer_step(x1, state)
            torch.cuda.synchronize()

    evs = [e for e in prof.key_averages() if ctime(e) > 0]
    total_us = sum(ctime(e) for e in evs) / STEPS
    n_launch = sum(e.count for e in evs) / STEPS
    evs.sort(key=ctime, reverse=True)
    print(f"infer_step single layer | per-step total {total_us:.2f} us | "
          f"{n_launch:.1f} kernel-launches/step")
    print(f"{'kernel':<50s} {'us/step':>9s} {'n/step':>7s}")
    for e in evs[:12]:
        print(f"{e.key[:48]:<50s} {ctime(e)/STEPS:9.2f} {e.count/STEPS:7.1f}")
    print(f"\n=> if total us/step is dominated by MANY small kernels each <a few us,"
          f"\n   it is launch-bound and a fused kernel will win.")
