"""
CUDA-accelerated TTT (LaCT fast-weight SwiGLU) operators for VLANeXt.

This package provides drop-in CUDA implementations of the two hot operators in
`src/models/ttt.py`:

  - causal_ttt_forward      <-> causal_block_fast_weight_swish_glu  (vision expert)
  - bidirectional_ttt_forward <-> fast_weight_swish_glu_weight_norm_mini_batch_apply (action expert; Phase 2+)

Design (see plans/ttt-cuda-kernel.plan.md):
  * GEMMs are batched 64x64 over (b * num_heads) and are already cuBLAS-optimal;
    we keep them as ATen `bmm` calls from C++.
  * The real cost in the eager torch path is the *kernel-launch storm* of ~20
    tiny elementwise / reduction ops per chunk per layer (silu, the gate/hidden
    products, silu_backprop, the Frobenius normalize, the weight-norm). Those are
    fused into a handful of custom CUDA kernels here.

The extension is JIT-compiled on first import via torch.utils.cpp_extension.load
(no pre-build step). If compilation or CUDA is unavailable, `HAVE_CUDA_TTT` is
False and callers should fall back to the pure-torch ops in `ttt.py`.

IMPORTANT (parity): with muon_update_steps == 0 the reference's
`zeropower_via_newtonschulz5` still divides each [d,d] gradient matrix by its
Frobenius norm (the NS loop is skipped, the normalize is NOT). The kernels
reproduce this. muon_update_steps > 0 (the Newton-Schulz orthogonalization) is
NOT yet supported by the CUDA path and falls back to torch.
"""

import os
import warnings

import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CSRC = os.path.join(_THIS_DIR, "csrc")

HAVE_CUDA_TTT = False
_EXT = None


def _load_extension():
    """JIT-compile and cache the CUDA extension. Returns the module or None."""
    global _EXT, HAVE_CUDA_TTT
    if _EXT is not None:
        return _EXT
    if not torch.cuda.is_available():
        return None
    try:
        from torch.utils.cpp_extension import load

        # Compile for the actual device capability (don't assume sm_90 just
        # because the path says h200 -- these machines are A800/sm_80). Allow
        # override via TORCH_CUDA_ARCH_LIST.
        if "TORCH_CUDA_ARCH_LIST" not in os.environ:
            major, minor = torch.cuda.get_device_capability()
            os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
        _EXT = load(
            name="ttt_fused_cuda",
            sources=[
                os.path.join(_CSRC, "ttt_fused.cpp"),
                os.path.join(_CSRC, "ttt_fused.cu"),
            ],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            extra_cflags=["-O3"],
            verbose=os.environ.get("TTT_CUDA_VERBOSE", "0") == "1",
        )
        HAVE_CUDA_TTT = True
        return _EXT
    except Exception as e:  # pragma: no cover - depends on toolchain
        warnings.warn(f"[ttt_cuda] JIT build failed, falling back to torch: {e}")
        _EXT = None
        HAVE_CUDA_TTT = False
        return None


def causal_ttt_forward(
    w0, w1, w2, q, k, v, lr0, lr1, lr2,
    chunk_size=256,
    vlm_k=None, vlm_v=None, vlm_lr0=None, vlm_lr1=None, vlm_lr2=None,
):
    """CUDA causal block fast-weight SwiGLU forward.

    Mirrors `causal_block_fast_weight_swish_glu` (muon_update_steps == 0 only).
    Shapes match the torch reference:
        w0,w2: [B, d_in, d_h]   w1: [B, d_h, d_out]
        q,k,v: [B, L, d]        lr0,lr1,lr2: [B, L, 1]
        vlm_*: [B, T_ctx, *] or None
    Returns (output [B, L, d_out], w0, w1, w2) -- updated fast weights.
    """
    ext = _load_extension()
    if ext is None:
        raise RuntimeError("ttt_cuda extension unavailable; use the torch fallback.")
    return ext.causal_ttt_forward(
        w0, w1, w2, q, k, v, lr0, lr1, lr2, chunk_size,
        vlm_k, vlm_v, vlm_lr0, vlm_lr1, vlm_lr2,
    )


class _CausalTTTFunction(torch.autograd.Function):
    """Autograd wrapper: CUDA forward + CUDA backward (Plan A).

    forward  -> ext.causal_ttt_forward  (CUDA, falls back to torch ref off-CUDA)
    backward -> ext.causal_ttt_backward (CUDA, Plan A fused BPTT) when available;
                else exact recompute (re-run torch reference under autograd).

    Both backends are validated bit-for-bit against the torch reference
    (test_ttt_cuda_backward.py) and the manual backward (test_ttt_manual_backward.py).
    """

    @staticmethod
    def forward(ctx, chunk_size, w0, w1, w2, q, k, v, lr0, lr1, lr2,
                vlm_k, vlm_v, vlm_lr0, vlm_lr1, vlm_lr2):
        from ..ttt import causal_block_fast_weight_swish_glu
        ctx.chunk_size = chunk_size
        ctx.save_for_backward(w0, w1, w2, q, k, v, lr0, lr1, lr2,
                              vlm_k, vlm_v, vlm_lr0, vlm_lr1, vlm_lr2)
        use_cuda = (HAVE_CUDA_TTT and q.is_cuda
                    and q.dtype in (torch.float32, torch.float16, torch.bfloat16))
        ctx.used_cuda = use_cuda
        if use_cuda:
            with torch.no_grad():
                out, nw0, nw1, nw2 = causal_ttt_forward(
                    w0, w1, w2, q, k, v, lr0, lr1, lr2,
                    chunk_size=chunk_size,
                    vlm_k=vlm_k, vlm_v=vlm_v,
                    vlm_lr0=vlm_lr0, vlm_lr1=vlm_lr1, vlm_lr2=vlm_lr2,
                )
        else:
            with torch.no_grad():
                out, nw0, nw1, nw2 = causal_block_fast_weight_swish_glu(
                    w0, w1, w2, q, k, v, lr0, lr1, lr2,
                    chunk_size=chunk_size, muon_update_steps=0,
                    vlm_k=vlm_k, vlm_v=vlm_v,
                    vlm_lr0=vlm_lr0, vlm_lr1=vlm_lr1, vlm_lr2=vlm_lr2,
                )
        return out, nw0, nw1, nw2

    @staticmethod
    def backward(ctx, g_out, g_w0, g_w1, g_w2):
        saved = ctx.saved_tensors
        (w0, w1, w2, q, k, v, lr0, lr1, lr2,
         vlm_k, vlm_v, vlm_lr0, vlm_lr1, vlm_lr2) = saved

        # ---- Plan A: CUDA fused backward ----
        ext = _load_extension()
        if ctx.used_cuda and ext is not None and hasattr(ext, "causal_ttt_backward"):
            g_out_c = g_out.contiguous()
            res = ext.causal_ttt_backward(
                w0, w1, w2, q, k, v, lr0, lr1, lr2, ctx.chunk_size,
                g_out_c, g_w0.contiguous(), g_w1.contiguous(), g_w2.contiguous(),
                vlm_k, vlm_v, vlm_lr0, vlm_lr1, vlm_lr2,
            )
            # res = [gw0,gw1,gw2, gq,gk,gv, glr0,glr1,glr2, (gvk,gvv,gvl0,gvl1,gvl2)]
            gw0, gw1, gw2, gq, gk, gv, gl0, gl1, gl2 = res[:9]
            if vlm_k is not None and len(res) >= 14:
                gvk, gvv, gvl0, gvl1, gvl2 = res[9:14]
            else:
                gvk = gvv = gvl0 = gvl1 = gvl2 = None
            # mask grads for inputs that didn't require grad
            def m(t, g):
                return g if (t is not None and t.requires_grad) else None
            return (None,  # chunk_size
                    m(w0, gw0), m(w1, gw1), m(w2, gw2),
                    m(q, gq), m(k, gk), m(v, gv),
                    m(lr0, gl0), m(lr1, gl1), m(lr2, gl2),
                    m(vlm_k, gvk), m(vlm_v, gvv),
                    m(vlm_lr0, gvl0), m(vlm_lr1, gvl1), m(vlm_lr2, gvl2))

        # ---- fallback: exact recompute backward ----
        from ..ttt import causal_block_fast_weight_swish_glu
        diff_inputs = [t for t in saved if t is not None and t.requires_grad]
        if not diff_inputs:
            return (None,) * 15
        with torch.enable_grad():
            ins = [t.detach().requires_grad_(t.requires_grad) if t is not None else None
                   for t in saved]
            (w0_, w1_, w2_, q_, k_, v_, lr0_, lr1_, lr2_,
             vk_, vv_, vl0_, vl1_, vl2_) = ins
            out, nw0, nw1, nw2 = causal_block_fast_weight_swish_glu(
                w0_, w1_, w2_, q_, k_, v_, lr0_, lr1_, lr2_,
                chunk_size=ctx.chunk_size, muon_update_steps=0,
                vlm_k=vk_, vlm_v=vv_, vlm_lr0=vl0_, vlm_lr1=vl1_, vlm_lr2=vl2_,
            )
            need = [t for t in ins if t is not None and t.requires_grad]
            grads = torch.autograd.grad(
                [out, nw0, nw1, nw2], need,
                grad_outputs=[g_out, g_w0, g_w1, g_w2], allow_unused=True,
            )
        gi = iter(grads)
        out_grads = [None]
        for t in ins:
            if t is not None and t.requires_grad:
                out_grads.append(next(gi))
            else:
                out_grads.append(None)
        return tuple(out_grads)


def causal_ttt(
    w0, w1, w2, q, k, v, lr0, lr1, lr2,
    chunk_size=256,
    vlm_k=None, vlm_v=None, vlm_lr0=None, vlm_lr1=None, vlm_lr2=None,
):
    """Autograd-aware causal TTT: CUDA forward + exact recompute backward.

    Drop-in for `causal_block_fast_weight_swish_glu(..., muon_update_steps=0)`.
    Returns (output, w0, w1, w2).
    """
    return _CausalTTTFunction.apply(
        chunk_size, w0, w1, w2, q, k, v, lr0, lr1, lr2,
        vlm_k, vlm_v, vlm_lr0, vlm_lr1, vlm_lr2,
    )

