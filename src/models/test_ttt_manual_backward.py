"""
L1..L5 parity test: manual backward (ttt_manual_backward) vs torch.autograd,
on the SAME forward recomputed with autograd at each complexity level.

Run:
  TORCHDYNAMO_DISABLE=1 PYTHONPATH="$PWD" \
    /mnt/afs-h200/yuyangcheng/venvs/fla_triton32/bin/python -m src.models.test_ttt_manual_backward
"""
import sys
import torch
import torch.nn.functional as F

from src.models.ttt_manual_backward import (
    chunk_forward, chunk_backward, silu_p, frobnorm_fwd, weightnorm_fwd,
)


# ---- an autograd reference forward, level-gated, matching chunk_forward exactly ----
def ref_forward(w0, w1, w2, q, k, v, lr0, lr1, lr2, w0n_t, w1n_t, w2n_t,
                chunk_size, level, vlm=None):
    def update_apply(w0, w1, w2, qi, ki, vi, l0, l1, l2, do_apply=True):
        oi = None
        if do_apply:
            oi = (F.silu(qi @ w0) * (qi @ w2)) @ w1
        gate = ki @ w0
        up = ki @ w2
        sg = F.silu(gate)
        hidden = sg * up
        dhidden = vi @ w1.transpose(-1, -2)
        dhid_bm = dhidden * sg
        dgate = dhidden * up
        dgba = dgate * silu_p(gate)
        raw_w1 = (hidden * l1).transpose(-1, -2) @ vi
        raw_w0 = (ki * l0).transpose(-1, -2) @ dgba
        raw_w2 = (ki * l2).transpose(-1, -2) @ dhid_bm
        if level >= 2:
            raw_w1 = raw_w1 / (raw_w1.flatten(1).norm(2, 1).view(-1, 1, 1) + 1e-7)
            raw_w0 = raw_w0 / (raw_w0.flatten(1).norm(2, 1).view(-1, 1, 1) + 1e-7)
            raw_w2 = raw_w2 / (raw_w2.flatten(1).norm(2, 1).view(-1, 1, 1) + 1e-7)
        w0p, w1p, w2p = w0 + raw_w0, w1 + raw_w1, w2 + raw_w2
        if level >= 3:
            w0p = w0p / (w0p.norm(2, 1, keepdim=True) + 1e-5) * w0n_t
            w1p = w1p / (w1p.norm(2, 1, keepdim=True) + 1e-5) * w1n_t
            w2p = w2p / (w2p.norm(2, 1, keepdim=True) + 1e-5) * w2n_t
        return oi, w0p, w1p, w2p

    if level >= 5 and vlm is not None:
        vk, vv, vl0, vl1, vl2 = vlm
        _, w0, w1, w2 = update_apply(w0, w1, w2, None, vk, vv, vl0, vl1, vl2, do_apply=False)

    L = q.shape[1]
    outs = []
    for s in range(0, L, chunk_size):
        e = min(s + chunk_size, L)
        oi, w0, w1, w2 = update_apply(
            w0, w1, w2, q[:, s:e], k[:, s:e], v[:, s:e],
            lr0[:, s:e], lr1[:, s:e], lr2[:, s:e])
        outs.append(oi)
    return torch.cat(outs, 1), w0, w1, w2


def manual_bptt(w0, w1, w2, q, k, v, lr0, lr1, lr2, w0n_t, w1n_t, w2n_t,
                chunk_size, level, g_out, g_w0n, g_w1n, g_w2n, vlm=None):
    """Forward (record caches) + reverse BPTT using chunk_forward/backward.
    If vlm is given (L5), prepend an update-only pre-update chunk."""
    L = q.shape[1]
    caches = []
    cw0, cw1, cw2 = w0, w1, w2

    vlm_cache = None
    if level >= 5 and vlm is not None:
        vk, vv, vl0, vl1, vl2 = vlm
        _, cw0, cw1, cw2, vlm_cache = chunk_forward(
            cw0, cw1, cw2, None, vk, vv, vl0, vl1, vl2,
            w0n_t, w1n_t, w2n_t, level, do_apply=False)

    starts = list(range(0, L, chunk_size))
    for s in starts:
        e = min(s + chunk_size, L)
        oi, cw0, cw1, cw2, cache = chunk_forward(
            cw0, cw1, cw2, q[:, s:e], k[:, s:e], v[:, s:e],
            lr0[:, s:e], lr1[:, s:e], lr2[:, s:e], w0n_t, w1n_t, w2n_t, level)
        caches.append((s, e, cache))

    g_q = torch.zeros_like(q); g_k = torch.zeros_like(k); g_v = torch.zeros_like(v)
    g_lr0 = torch.zeros_like(lr0); g_lr1 = torch.zeros_like(lr1); g_lr2 = torch.zeros_like(lr2)
    gw0, gw1, gw2 = g_w0n.clone(), g_w1n.clone(), g_w2n.clone()

    for (s, e, cache) in reversed(caches):
        g_oi = g_out[:, s:e]
        dw0, dw1, dw2, dq, dk, dv, dl0, dl1, dl2 = chunk_backward(
            g_oi, gw0, gw1, gw2, cache)
        g_q[:, s:e] += dq; g_k[:, s:e] += dk; g_v[:, s:e] += dv
        g_lr0[:, s:e] += dl0; g_lr1[:, s:e] += dl1; g_lr2[:, s:e] += dl2
        gw0, gw1, gw2 = dw0, dw1, dw2

    out = dict(w0=gw0, w1=gw1, w2=gw2, q=g_q, k=g_k, v=g_v,
               lr0=g_lr0, lr1=g_lr1, lr2=g_lr2)
    if vlm_cache is not None:
        # pre-update: update-only, g_oi unused (None apply). Output weight-state
        # grad (gw0..) flows into it; produces grads on vk/vv/vlr and the
        # ORIGINAL w0/w1/w2 entering the pre-update.
        dw0, dw1, dw2, _dq, dvk, dvv, dvl0, dvl1, dvl2 = chunk_backward(
            None, gw0, gw1, gw2, vlm_cache)
        out["w0"], out["w1"], out["w2"] = dw0, dw1, dw2
        out["vk"], out["vv"] = dvk, dvv
        out["vl0"], out["vl1"], out["vl2"] = dvl0, dvl1, dvl2
    return out


def _mk(B, L, d, dh, T, dev, seed):
    g = torch.Generator(device=dev).manual_seed(seed)
    rk = lambda *s: torch.randn(*s, generator=g, device=dev, dtype=torch.float64)
    pr = lambda *s: torch.rand(*s, generator=g, device=dev, dtype=torch.float64) * 0.02 + 0.001
    return dict(
        w0=rk(B, d, dh), w1=rk(B, dh, d), w2=rk(B, d, dh),
        q=rk(B, L, d), k=rk(B, L, d), v=rk(B, L, d),
        lr0=pr(B, L, 1), lr1=pr(B, L, 1), lr2=pr(B, L, 1),
        vk=rk(B, T, d), vv=rk(B, T, d), vl0=pr(B, T, 1), vl1=pr(B, T, 1), vl2=pr(B, T, 1))


def main():
    if not torch.cuda.is_available():
        print("SKIP: no CUDA"); return 0
    dev = "cuda"
    B, L, d, dh, T = 3, 32, 8, 8, 5
    results = []
    for level in (1, 2, 3, 4, 5):
        cs = 32 if level < 4 else 8   # L1-3 single chunk; L4-5 multi-chunk
        t = _mk(B, L, d, dh, T, dev, seed=10 + level)
        # detached target norms (computed once from initial weights, like the op)
        w0n_t = t["w0"].norm(2, 1, keepdim=True).detach()
        w1n_t = t["w1"].norm(2, 1, keepdim=True).detach()
        w2n_t = t["w2"].norm(2, 1, keepdim=True).detach()

        # ---- autograd reference ----
        ref_keys = ("w0", "w1", "w2", "q", "k", "v", "lr0", "lr1", "lr2")
        if level >= 5:
            ref_keys = ref_keys + ("vk", "vv", "vl0", "vl1", "vl2")
        leaves = {kk: t[kk].clone().requires_grad_(True) for kk in ref_keys}
        vlm = ((leaves["vk"], leaves["vv"], leaves["vl0"], leaves["vl1"], leaves["vl2"])
               if level >= 5 else None)
        out, fw0, fw1, fw2 = ref_forward(
            leaves["w0"], leaves["w1"], leaves["w2"], leaves["q"], leaves["k"], leaves["v"],
            leaves["lr0"], leaves["lr1"], leaves["lr2"], w0n_t, w1n_t, w2n_t, cs, level, vlm)
        # random upstream grads on output AND final weights (full vjp coverage)
        g = torch.Generator(device=dev).manual_seed(99)
        g_out = torch.randn(*out.shape, generator=g, device=dev, dtype=torch.float64)
        g_fw0 = torch.randn(*fw0.shape, generator=g, device=dev, dtype=torch.float64)
        g_fw1 = torch.randn(*fw1.shape, generator=g, device=dev, dtype=torch.float64)
        g_fw2 = torch.randn(*fw2.shape, generator=g, device=dev, dtype=torch.float64)
        grads = torch.autograd.grad(
            [out, fw0, fw1, fw2], [leaves[kk] for kk in ref_keys],
            grad_outputs=[g_out, g_fw0, g_fw1, g_fw2])
        ref = dict(zip(ref_keys, grads))

        # ---- manual backward ----
        man = manual_bptt(
            t["w0"], t["w1"], t["w2"], t["q"], t["k"], t["v"],
            t["lr0"], t["lr1"], t["lr2"], w0n_t, w1n_t, w2n_t, cs, level,
            g_out, g_fw0, g_fw1, g_fw2,
            vlm=((t["vk"], t["vv"], t["vl0"], t["vl1"], t["vl2"]) if level >= 5 else None))

        max_err = 0.0
        worst = ""
        for kk in ref_keys:
            e = (ref[kk] - man[kk]).abs().max().item()
            if e > max_err:
                max_err, worst = e, kk
        ok = max_err < 1e-8
        print(f"[{'OK ' if ok else 'BAD'}] L{level} (cs={cs}) | max grad err={max_err:.2e} "
              f"(worst={worst}) (tol=1e-8, fp64)")
        results.append(ok)

    allok = all(results)
    print("\n=== MANUAL BACKWARD L1-L5", "PASS ===" if allok else "FAIL ===")
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
