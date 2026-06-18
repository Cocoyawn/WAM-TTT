"""
Localize the Plan-A CUDA backward bug: compare ext.causal_ttt_backward against
the proven torch manual_bptt (matches autograd to 1e-12), per-output-tensor,
fp32, multi-chunk. Prints which grad (w0/w1/w2/q/k/v/lr) diverges.

Run:
  TORCHDYNAMO_DISABLE=1 PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES=5 \
    /mnt/afs-h200/yuyangcheng/venvs/fla_triton32/bin/python -m src.models.test_ttt_cuda_backward_localize
"""
import sys
import torch

from src.models import ttt_cuda
from src.models.test_ttt_manual_backward import manual_bptt


def main():
    if not torch.cuda.is_available():
        print("SKIP: no CUDA"); return 0
    ext = ttt_cuda._load_extension()
    if ext is None or not hasattr(ext, "causal_ttt_backward"):
        print("FAIL: backward not built"); return 1
    dev = "cuda"
    dt = torch.float32
    B, L, d, dh = 4, 128, 16, 16
    g = torch.Generator(device=dev).manual_seed(3)
    rk = lambda *s: torch.randn(*s, generator=g, device=dev, dtype=dt)
    pr = lambda *s: torch.rand(*s, generator=g, device=dev, dtype=dt) * 0.02 + 0.001
    w0 = rk(B, d, dh); w1 = rk(B, dh, d); w2 = rk(B, d, dh)
    q = rk(B, L, d); k = rk(B, L, d); v = rk(B, L, d)
    lr0 = pr(B, L, 1); lr1 = pr(B, L, 1); lr2 = pr(B, L, 1)
    w0n = w0.norm(2, 1, True); w1n = w1.norm(2, 1, True); w2n = w2.norm(2, 1, True)

    g_out = rk(B, L, d)
    g_fw0 = rk(B, d, dh); g_fw1 = rk(B, dh, d); g_fw2 = rk(B, d, dh)

    for cs in (256, 64, 32):
        # torch manual (ground truth)
        man = manual_bptt(w0, w1, w2, q, k, v, lr0, lr1, lr2, w0n, w1n, w2n,
                          cs, level=3, g_out=g_out, g_w0n=g_fw0, g_w1n=g_fw1, g_w2n=g_fw2)
        # cuda
        res = ext.causal_ttt_backward(
            w0, w1, w2, q, k, v, lr0, lr1, lr2, cs,
            g_out, g_fw0, g_fw1, g_fw2, None, None, None, None, None)
        cu = dict(zip(("w0", "w1", "w2", "q", "k", "v", "lr0", "lr1", "lr2"), res[:9]))
        print(f"--- cs={cs} ({(L + cs - 1)//cs} chunks) ---")
        for kk in ("w0", "w1", "w2", "q", "k", "v", "lr0", "lr1", "lr2"):
            e = (man[kk].float() - cu[kk].float()).abs().max().item()
            mag = man[kk].float().abs().max().item()
            rel = e / (mag + 1e-9)
            print(f"    {kk:4s}: abs={e:.2e} rel={rel:.2e} (mag={mag:.2e}) {'<<<' if rel > 1e-2 else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
