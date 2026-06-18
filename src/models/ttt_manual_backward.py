"""
Manual (autograd-free) backward for the causal TTT fast-weight operator.

This is the MATH GROUND TRUTH for the Plan-A CUDA backward kernel: each forward
primitive is paired with its explicit vjp, chained in reverse, and validated
against torch.autograd at progressive complexity levels (L1..L5). Once this
matches autograd, it is ported function-by-function to CUDA (Stage 2), using
THIS as the oracle.

muon_update_steps == 0 only (zeropower reduces to Frobenius-normalize).

Levels (toggle via `level`):
  L1: single chunk, raw update (w_new = w + raw_grad), no norms
  L2: + Frobenius-normalize on each raw grad
  L3: + weight-norm rescale (detached target norm) on each w_new
  L4: + multi-chunk BPTT loop
  L5: + global (non-causal) VLM pre-update
"""
import torch
import torch.nn.functional as F


# ---------------- forward primitives + vjps ----------------

def _sigmoid(x):
    return torch.sigmoid(x)


def silu_p(x):
    """silu'(x) = s + x*s*(1-s) = s*(1 + x*(1-s)), s=sigmoid(x)."""
    s = _sigmoid(x)
    return s * (1.0 + x * (1.0 - s))


def silu_pp(x):
    """silu''(x) = s(1-s)*(2 + x*(1-2s)), s=sigmoid(x)."""
    s = _sigmoid(x)
    s1 = s * (1.0 - s)
    return s1 * (2.0 + x * (1.0 - 2.0 * s))


def frobnorm_fwd(x, eps=1e-7):
    """y = x / (||x||_F + eps), Frobenius over dims (1,2) per batch. Returns (y, cache)."""
    n = x.flatten(1).norm(2, dim=1).view(-1, *([1] * (x.dim() - 1)))  # [b,1,1]
    r = 1.0 / (n + eps)
    return x * r, (x, n, r)


def frobnorm_bwd(gy, cache):
    """gx = r*(gy - (r/n)*<gy,x>*x)."""
    x, n, r = cache
    dot = (gy * x).flatten(1).sum(1).view(-1, *([1] * (x.dim() - 1)))  # [b,1,1]
    return r * (gy - (r / n) * dot * x)


def weightnorm_fwd(w_pre, w_target_norm, eps=1e-5):
    """y = w_pre/(||w_pre||_dim1 + eps) * w_target_norm. norm over dim=1 keepdim.
    w_target_norm is DETACHED (no grad). Returns (y, cache)."""
    nc = w_pre.norm(2, dim=1, keepdim=True)  # [b,1,C]
    rc = 1.0 / (nc + eps)
    return w_pre * rc * w_target_norm, (w_pre, nc, rc, w_target_norm)


def weightnorm_bwd(gy, cache):
    """per-column: gx = wn*( rc*gy - (rc^2/nc)*<gy,w_pre>_dim1 * w_pre )."""
    w_pre, nc, rc, wn = cache
    dotc = (gy * w_pre).sum(dim=1, keepdim=True)  # [b,1,C]
    return wn * (rc * gy - (rc * rc / nc) * dotc * w_pre)


# ---------------- the per-chunk forward (records a cache) ----------------

def chunk_forward(w0, w1, w2, qi, ki, vi, lr0i, lr1i, lr2i, w0n_t, w1n_t, w2n_t,
                  level, do_apply=True):
    """One chunk: apply (old w) -> oi ; update (k,v) -> new w. Returns (oi, w0n,w1n,w2n, cache).

    do_apply=False is the VLM global pre-update (update-only, no query apply); oi is None.
    """
    # ----- apply (uses OLD weights) -----
    if do_apply:
        gate_q = qi @ w0
        up_q = qi @ w2
        sq = F.silu(gate_q)
        h_q = sq * up_q
        oi = h_q @ w1
    else:
        gate_q = up_q = sq = h_q = oi = None

    # ----- update -----
    gate = ki @ w0
    up = ki @ w2
    sg = F.silu(gate)
    hidden = sg * up
    dhidden = vi @ w1.transpose(-1, -2)
    dhid_bm = dhidden * sg
    dgate = dhidden * up
    m = silu_p(gate)
    dgba = dgate * m  # silu_backprop(dgate, gate)

    A1 = hidden * lr1i
    A0 = ki * lr0i
    A2 = ki * lr2i
    raw_w1 = A1.transpose(-1, -2) @ vi
    raw_w0 = A0.transpose(-1, -2) @ dgba
    raw_w2 = A2.transpose(-1, -2) @ dhid_bm

    fc0 = fc1 = fc2 = None
    if level >= 2:
        raw_w1, fc1 = frobnorm_fwd(raw_w1)
        raw_w0, fc0 = frobnorm_fwd(raw_w0)
        raw_w2, fc2 = frobnorm_fwd(raw_w2)

    w0_pre = w0 + raw_w0
    w1_pre = w1 + raw_w1
    w2_pre = w2 + raw_w2

    wc0 = wc1 = wc2 = None
    if level >= 3:
        w0n, wc0 = weightnorm_fwd(w0_pre, w0n_t)
        w1n, wc1 = weightnorm_fwd(w1_pre, w1n_t)
        w2n, wc2 = weightnorm_fwd(w2_pre, w2n_t)
    else:
        w0n, w1n, w2n = w0_pre, w1_pre, w2_pre

    cache = dict(
        w0=w0, w1=w1, w2=w2, qi=qi, ki=ki, vi=vi, lr0i=lr0i, lr1i=lr1i, lr2i=lr2i,
        gate_q=gate_q, up_q=up_q, sq=sq, h_q=h_q,
        gate=gate, up=up, sg=sg, hidden=hidden, dhidden=dhidden, dhid_bm=dhid_bm,
        dgate=dgate, m=m, A0=A0, A1=A1, A2=A2,
        fc0=fc0, fc1=fc1, fc2=fc2, wc0=wc0, wc1=wc1, wc2=wc2, level=level,
        do_apply=do_apply,
    )
    return oi, w0n, w1n, w2n, cache


def chunk_backward(g_oi, g_w0n, g_w1n, g_w2n, cache):
    """vjp of chunk_forward. Returns grads on (w0,w1,w2,qi,ki,vi,lr0i,lr1i,lr2i)
    where (w0,w1,w2) grads are wrt the OLD weights entering this chunk."""
    c = cache
    level = c["level"]
    w0, w1, w2 = c["w0"], c["w1"], c["w2"]
    qi, ki, vi = c["qi"], c["ki"], c["vi"]
    lr0i, lr1i, lr2i = c["lr0i"], c["lr1i"], c["lr2i"]

    g_w0 = torch.zeros_like(w0); g_w1 = torch.zeros_like(w1); g_w2 = torch.zeros_like(w2)
    g_qi = None if qi is None else torch.zeros_like(qi)
    g_ki = torch.zeros_like(ki); g_vi = torch.zeros_like(vi)

    # ---- weightnorm vjp: g on w_pre ----
    if level >= 3:
        g_w0_pre = weightnorm_bwd(g_w0n, c["wc0"])
        g_w1_pre = weightnorm_bwd(g_w1n, c["wc1"])
        g_w2_pre = weightnorm_bwd(g_w2n, c["wc2"])
    else:
        g_w0_pre, g_w1_pre, g_w2_pre = g_w0n, g_w1n, g_w2n

    # w_pre = w_old + raw -> g_w_old += g_w_pre ; g_raw = g_w_pre
    g_w0 += g_w0_pre; g_w1 += g_w1_pre; g_w2 += g_w2_pre
    g_raw_w0, g_raw_w1, g_raw_w2 = g_w0_pre, g_w1_pre, g_w2_pre

    # ---- frobnorm vjp ----
    if level >= 2:
        g_raw_w0 = frobnorm_bwd(g_raw_w0, c["fc0"])
        g_raw_w1 = frobnorm_bwd(g_raw_w1, c["fc1"])
        g_raw_w2 = frobnorm_bwd(g_raw_w2, c["fc2"])

    # ---- raw_w1 = A1^T @ vi ----
    A1 = c["A1"]
    g_A1 = vi @ g_raw_w1.transpose(-1, -2)
    g_vi += A1 @ g_raw_w1
    # A1 = hidden*lr1
    g_hidden = g_A1 * lr1i
    g_lr1 = (g_A1 * c["hidden"]).sum(-1, keepdim=True)

    # ---- raw_w0 = A0^T @ dgba ----
    A0 = c["A0"]
    dgba = c["dgate"] * c["m"]
    g_A0 = dgba @ g_raw_w0.transpose(-1, -2)
    g_dgba = A0 @ g_raw_w0
    g_ki += g_A0 * lr0i
    g_lr0 = (g_A0 * ki).sum(-1, keepdim=True)

    # ---- raw_w2 = A2^T @ dhid_bm ----
    A2 = c["A2"]
    g_A2 = c["dhid_bm"] @ g_raw_w2.transpose(-1, -2)
    g_dhid_bm = A2 @ g_raw_w2
    g_ki += g_A2 * lr2i
    g_lr2 = (g_A2 * ki).sum(-1, keepdim=True)

    # ---- dgba = dgate * m(gate) ; m=silu'(gate) ----
    g_dgate = g_dgba * c["m"]
    g_gate = g_dgba * c["dgate"] * silu_pp(c["gate"])

    # ---- dgate = dhidden * up ----
    g_dhidden = g_dgate * c["up"]
    g_up = g_dgate * c["dhidden"]

    # ---- dhid_bm = dhidden * sg ----
    g_dhidden = g_dhidden + g_dhid_bm * c["sg"]
    g_sg = g_dhid_bm * c["dhidden"]

    # ---- dhidden = vi @ w1^T ----
    g_vi += g_dhidden @ w1
    g_w1 += g_dhidden.transpose(-1, -2) @ vi

    # ---- hidden = sg * up ----
    g_sg = g_sg + g_hidden * c["up"]
    g_up = g_up + g_hidden * c["sg"]

    # ---- up = ki @ w2 ----
    g_ki += g_up @ w2.transpose(-1, -2)
    g_w2 += ki.transpose(-1, -2) @ g_up

    # ---- sg = silu(gate) ----
    g_gate = g_gate + g_sg * silu_p(c["gate"])

    # ---- gate = ki @ w0 ----
    g_ki += g_gate @ w0.transpose(-1, -2)
    g_w0 += ki.transpose(-1, -2) @ g_gate

    # ======== apply path (uses OLD weights) ========
    if c["do_apply"]:
        h_q = c["h_q"]
        g_h_q = g_oi @ w1.transpose(-1, -2)
        g_w1 += h_q.transpose(-1, -2) @ g_oi
        # h_q = sq * up_q
        g_sq = g_h_q * c["up_q"]
        g_up_q = g_h_q * c["sq"]
        # up_q = qi @ w2
        g_qi += g_up_q @ w2.transpose(-1, -2)
        g_w2 += qi.transpose(-1, -2) @ g_up_q
        # sq = silu(gate_q)
        g_gate_q = g_sq * silu_p(c["gate_q"])
        # gate_q = qi @ w0
        g_qi += g_gate_q @ w0.transpose(-1, -2)
        g_w0 += qi.transpose(-1, -2) @ g_gate_q

    return g_w0, g_w1, g_w2, g_qi, g_ki, g_vi, g_lr0, g_lr1, g_lr2
