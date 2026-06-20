"""
Test-Time Training (TTT / LaCT-style) fast-weight layer for VLANeXt.

Copied and adapted from ZipMap (`zipmap/layers/ttt.py`), which itself is a
derivative of LaCT (https://arxiv.org/abs/2505.23884).

This file provides a SwiGLU fast-weight TTT operator in two flavors, so that it
can *semantically replace* the two different attention blocks in VLANeXt:

  - bidirectional  -> replaces the action-expert `MoEBlock` (no mask, the 8
                      action tokens attend to each other bidirectionally because
                      they are denoised jointly by the diffusion head).
  - causal         -> replaces the vision-expert `MoEGeneratorBlock` (image
                      tokens are generated autoregressively and must keep the
                      image->image causal ordering).

NOTE: This module only provides the self-TTT operators + the multihead wrapper.
Wiring it into `policies.py` / `generator.py` (the [A,A,A,T] interleave and the
VLM-hidden-state injection) is a separate integration step.

Adaptations vs. the ZipMap source:
  1. The fused-triton-kernel import is made optional (VLANeXt does not ship the
     triton kernels); `use_fused_kernels=True` raises if they are unavailable.
  2. A `causal_block_fast_weight_swish_glu` operator (LaCT-style shifted
     block-causal, apply-then-update) is added.
  3. `FastWeightGluMLPMultihead` gains a `causal` flag selecting the operator.
"""

import collections
import math

import torch
from torch import nn

import torch.nn.functional as F
from einops import rearrange

# Optional fused triton kernels (not shipped with VLANeXt). Only required when
# `use_fused_kernels=True`. Kept as a soft dependency so the import never breaks.
try:
    from .lact_with_act_ckpt_plain import (
        lact_swiglu_ffn_fast_weight_grads_with_ckpt,
        fused_swiglu_ffn_fwd_with_ckpt,
    )
    _FUSED_KERNELS_AVAILABLE = True
except ImportError:
    lact_swiglu_ffn_fast_weight_grads_with_ckpt = None
    fused_swiglu_ffn_fwd_with_ckpt = None
    _FUSED_KERNELS_AVAILABLE = False


TTTOperator = collections.namedtuple("TTTOperator", ["start", "end", "update", "apply"])


# `nn.RMSNorm` only exists in torch >= 2.4. Provide a fallback for older torch.
if hasattr(nn, "RMSNorm"):
    RMSNorm = nn.RMSNorm
else:
    class RMSNorm(nn.Module):
        def __init__(self, dim, eps=1e-5, elementwise_affine=True):
            super().__init__()
            self.eps = eps
            self.elementwise_affine = elementwise_affine
            if elementwise_affine:
                self.weight = nn.Parameter(torch.ones(dim))
            else:
                self.register_parameter("weight", None)

        def forward(self, x):
            dtype = x.dtype
            x = x.float()
            x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
            x = x.to(dtype)
            if self.weight is not None:
                x = x * self.weight
            return x


@torch.compile
def inv_softplus(x):
    y = x + math.log(-math.expm1(-x))
    return y

@torch.compile
def silu_backprop(dy: torch.Tensor, x: torch.Tensor):
    """
    Args:
        dy: [b, d, l], gradient of the outer loss wrt the y
        x: [b, d, l], input of the silu activation
    outs:
        dx: [b, d, l], gradient of the outer loss wrt the x
        dx = dy * sigma * (1 + x * (1 - sigma))
    """
    sigma = torch.sigmoid(x)
    dx = dy * sigma * (1 + x * (1 - sigma))
    return dx

@torch.compile()
def zeropower_via_newtonschulz5(G, steps):
    """
    modified from https://github.com/MoonshotAI/Moonlight/blob/master/examples/toy_train.py#L49
    Major change: G is [b, d, d] rather than [d, d]
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    Args:
        G: [b, d, d]
        steps: int
    Returns:
        X: [b, d, d]
    """
    assert len(G.shape) == 3
    a, b, c = (3.4445, -4.7750, 2.0315)
    # X = G.bfloat16()
    X = G.to(dtype=torch.bfloat16, device=G.device).contiguous()
    if G.size(1) > G.size(2):
        X = X.transpose(1, 2)
    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(1, 2), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.transpose(1, 2)
        B = (
            b * A + c * A @ A
        )  # adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(1) > G.size(2):
        X = X.transpose(1, 2)
    return X


@torch.compile(dynamic=True)
def fast_weight_swish_glu_weight_norm_mini_batch_apply(
    w0: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lr0: torch.Tensor,
    lr1: torch.Tensor,
    lr2: torch.Tensor,
    ttt_ua_order: list,
    muon_update_steps: int = 0,
):
    """
    Bidirectional SwiGLU fast-weight TTT operator (driven by an explicit
    update/apply order list `ttt_ua_order`).

    Note:
    Forward:
    (silu(x @ w0) * (x @ w2)) @ w1

    w0, w2: [b, d, dh]
    w1:     [b, dh, d]
    q: [b, l, d]
    k: [b, l, d]
    v: [b, l, d]
    lr0, lr1, lr2: [b, l, 1]
    """
    w0_norm = w0.detach().norm(dim=1, keepdim=True)
    w1_norm = w1.detach().norm(dim=1, keepdim=True)
    w2_norm = w2.detach().norm(dim=1, keepdim=True)

    output = []
    for start, end, update, apply in ttt_ua_order:
        w0_now, w1_now, w2_now = w0, w1, w2
        # all tokens
        if end == -1:
            end = q.shape[1]


        if update:
            ki, vi = k[:, start:end, :], v[:, start:end, :]  # bf16
            lr0i = lr0[:, start:end, :]  # [b, l, d/1] fp32
            lr1i = lr1[:, start:end, :]  # [b, l, d/1] fp32
            lr2i = lr2[:, start:end, :]  # [b, l, d/1] fp32

            gate_before_act = ki @ w0_now       # b[b, l, dh] = [b, l, d] @ [b, d, dh]
            hidden_before_mul = ki @ w2_now     # b[b, l, dh] = [b, l, d] @ [b, d, dh]
            hidden = F.silu(gate_before_act, inplace=False) * hidden_before_mul

            dhidden = vi @ w1_now.transpose(-1, -2)  # [b, l, dh] = [b, l, d] @ [b, d, dh]
            dhidden_before_mul = dhidden * F.silu(gate_before_act, inplace=False)
            dgate = dhidden * hidden_before_mul
            dgate_before_act = silu_backprop(dgate, gate_before_act)

            # [b, dh, l] @ [b, l, d] -> [b, dh, d]
            # cast lr-scaled tensors back to operand dtype (lr is fp32; matches LaCT's .type_as)
            w1_grad = zeropower_via_newtonschulz5(
                (hidden * lr1i).to(vi.dtype).transpose(-1, -2) @ vi, muon_update_steps
            )
            w0_grad = zeropower_via_newtonschulz5(
                (ki * lr0i).to(dgate_before_act.dtype).transpose(-1, -2) @ dgate_before_act, muon_update_steps
            )
            w2_grad = zeropower_via_newtonschulz5(
                (ki * lr2i).to(dhidden_before_mul.dtype).transpose(-1, -2) @ dhidden_before_mul, muon_update_steps
            )


            w1_now = w1_now + w1_grad
            w0_now = w0_now + w0_grad
            w2_now = w2_now + w2_grad


            # do weight norm here
            w0_now = w0_now / (w0_now.norm(dim=1, keepdim=True) + 1e-5) * w0_norm
            w1_now = w1_now / (w1_now.norm(dim=1, keepdim=True) + 1e-5) * w1_norm
            w2_now = w2_now / (w2_now.norm(dim=1, keepdim=True) + 1e-5) * w2_norm


            w0, w1, w2 = w0_now, w1_now, w2_now

        if apply:
            # Only calculate the output in the last repeat.
            qi = q[:, start:end, :]
            oi = (F.silu(qi @ w0_now, inplace=True) * (qi @ w2_now)) @ w1_now
            output.append(oi)

    output = torch.cat(output, dim=1)

    return output, w0, w1, w2


def _vlm_preupdate(w0, w1, w2, vlm_k, vlm_v, vlm_lr0, vlm_lr1, vlm_lr2):
    """The global (non-causal) VLM pre-update, muon_update_steps=0.

    Extracted from causal_block_fast_weight_swish_glu's vlm_k branch so the
    incremental inference path can build the cached fast weights identically.
    Returns the post-pre-update (w0, w1, w2).
    """
    w0_norm = w0.detach().norm(dim=1, keepdim=True)
    w1_norm = w1.detach().norm(dim=1, keepdim=True)
    w2_norm = w2.detach().norm(dim=1, keepdim=True)

    gate_before_act = vlm_k @ w0
    hidden_before_mul = vlm_k @ w2
    hidden = F.silu(gate_before_act, inplace=False) * hidden_before_mul

    dhidden = vlm_v @ w1.transpose(-1, -2)
    dhidden_before_mul = dhidden * F.silu(gate_before_act, inplace=False)
    dgate = dhidden * hidden_before_mul
    dgate_before_act = silu_backprop(dgate, gate_before_act)

    w1_grad = zeropower_via_newtonschulz5(
        (hidden * vlm_lr1).to(vlm_v.dtype).transpose(-1, -2) @ vlm_v, 0)
    w0_grad = zeropower_via_newtonschulz5(
        (vlm_k * vlm_lr0).to(dgate_before_act.dtype).transpose(-1, -2) @ dgate_before_act, 0)
    w2_grad = zeropower_via_newtonschulz5(
        (vlm_k * vlm_lr2).to(dhidden_before_mul.dtype).transpose(-1, -2) @ dhidden_before_mul, 0)

    w0 = w0 + w0_grad
    w1 = w1 + w1_grad
    w2 = w2 + w2_grad

    w0 = w0 / (w0.norm(dim=1, keepdim=True) + 1e-5) * w0_norm
    w1 = w1 / (w1.norm(dim=1, keepdim=True) + 1e-5) * w1_norm
    w2 = w2 / (w2.norm(dim=1, keepdim=True) + 1e-5) * w2_norm
    return w0, w1, w2


@torch.compile(dynamic=True)
def causal_block_fast_weight_swish_glu(
    w0: torch.Tensor,  # [b, d, dh]
    w1: torch.Tensor,  # [b, dh, d]
    w2: torch.Tensor,  # [b, d, dh]
    q: torch.Tensor,  # [b, l, d]
    k: torch.Tensor,  # [b, l, d]
    v: torch.Tensor,  # [b, l, d]
    lr0: torch.Tensor,  # [b, l, 1]
    lr1: torch.Tensor,  # [b, l, 1]
    lr2: torch.Tensor,  # [b, l, 1]
    chunk_size: int = 64,
    muon_update_steps: int = 0,
    vlm_k: torch.Tensor = None,   # [b, t_vlm, d]  global (non-causal) context
    vlm_v: torch.Tensor = None,   # [b, t_vlm, d]
    vlm_lr0: torch.Tensor = None,  # [b, t_vlm, 1]
    vlm_lr1: torch.Tensor = None,  # [b, t_vlm, 1]
    vlm_lr2: torch.Tensor = None,  # [b, t_vlm, 1]
):
    """
    Shifted block-causal SwiGLU fast-weight TTT operator (LaCT causal style).

    Semantics: APPLY-then-UPDATE per chunk.
      For chunk i: first produce the output for q_i using the *current* fast
      weights (which were trained only on chunks < i), then update the fast
      weights with (k_i, v_i). This guarantees block-level causality -- a query
      in chunk i never sees keys from chunk i or later -- matching the
      image->image causal mask of `MoEGeneratorBlock`.

    Optional global VLM context (vlm_k/vlm_v/vlm_lr*): when provided, the fast
    weights are FIRST updated once with the VLM key/value tokens (non-causally,
    i.e. before any image chunk), so every image query can read the VLM context.
    This matches `MoEGeneratorBlock`'s mask where image->image is causal but
    image->VLM is fully visible.

    Use `chunk_size=1` for strict token-level causality (slow); larger chunks
    trade exactness for speed, exactly like LaCT's block-causal variant.

    Same forward as the bidirectional op: (silu(x @ w0) * (x @ w2)) @ w1
    """
    w0_norm = w0.detach().norm(dim=1, keepdim=True)
    w1_norm = w1.detach().norm(dim=1, keepdim=True)
    w2_norm = w2.detach().norm(dim=1, keepdim=True)

    # ----- global (non-causal) VLM context pre-update: makes VLM fully visible
    if vlm_k is not None:
        gate_before_act = vlm_k @ w0
        hidden_before_mul = vlm_k @ w2
        hidden = F.silu(gate_before_act, inplace=False) * hidden_before_mul

        dhidden = vlm_v @ w1.transpose(-1, -2)
        dhidden_before_mul = dhidden * F.silu(gate_before_act, inplace=False)
        dgate = dhidden * hidden_before_mul
        dgate_before_act = silu_backprop(dgate, gate_before_act)

        w1_grad = zeropower_via_newtonschulz5(
            (hidden * vlm_lr1).to(vlm_v.dtype).transpose(-1, -2) @ vlm_v, muon_update_steps
        )
        w0_grad = zeropower_via_newtonschulz5(
            (vlm_k * vlm_lr0).to(dgate_before_act.dtype).transpose(-1, -2) @ dgate_before_act, muon_update_steps
        )
        w2_grad = zeropower_via_newtonschulz5(
            (vlm_k * vlm_lr2).to(dhidden_before_mul.dtype).transpose(-1, -2) @ dhidden_before_mul, muon_update_steps
        )

        w0 = w0 + w0_grad
        w1 = w1 + w1_grad
        w2 = w2 + w2_grad

        w0 = w0 / (w0.norm(dim=1, keepdim=True) + 1e-5) * w0_norm
        w1 = w1 / (w1.norm(dim=1, keepdim=True) + 1e-5) * w1_norm
        w2 = w2 / (w2.norm(dim=1, keepdim=True) + 1e-5) * w2_norm

    seq_len = q.shape[1]
    output = []
    for s_index in range(0, seq_len, chunk_size):
        e_index = min(s_index + chunk_size, seq_len)

        ######### apply current (causal) fast weights to this chunk's query #########
        qi = q[:, s_index:e_index, :]
        oi = (F.silu(qi @ w0, inplace=False) * (qi @ w2)) @ w1
        output.append(oi)

        ######### then update the fast weights with this chunk's (k, v) #########
        ki, vi = k[:, s_index:e_index, :], v[:, s_index:e_index, :]
        lr0i = lr0[:, s_index:e_index, :]
        lr1i = lr1[:, s_index:e_index, :]
        lr2i = lr2[:, s_index:e_index, :]

        gate_before_act = ki @ w0
        hidden_before_mul = ki @ w2
        hidden = F.silu(gate_before_act, inplace=False) * hidden_before_mul

        dhidden = vi @ w1.transpose(-1, -2)
        dhidden_before_mul = dhidden * F.silu(gate_before_act, inplace=False)
        dgate = dhidden * hidden_before_mul
        dgate_before_act = silu_backprop(dgate, gate_before_act)

        w1_grad = zeropower_via_newtonschulz5(
            (hidden * lr1i).to(vi.dtype).transpose(-1, -2) @ vi, muon_update_steps
        )
        w0_grad = zeropower_via_newtonschulz5(
            (ki * lr0i).to(dgate_before_act.dtype).transpose(-1, -2) @ dgate_before_act, muon_update_steps
        )
        w2_grad = zeropower_via_newtonschulz5(
            (ki * lr2i).to(dhidden_before_mul.dtype).transpose(-1, -2) @ dhidden_before_mul, muon_update_steps
        )

        w0 = w0 + w0_grad
        w1 = w1 + w1_grad
        w2 = w2 + w2_grad

        w0 = w0 / (w0.norm(dim=1, keepdim=True) + 1e-5) * w0_norm
        w1 = w1 / (w1.norm(dim=1, keepdim=True) + 1e-5) * w1_norm
        w2 = w2 / (w2.norm(dim=1, keepdim=True) + 1e-5) * w2_norm

    output = torch.cat(output, dim=1)
    return output, w0, w1, w2


@torch.compile(dynamic=True)
def bidirectional_lact_swiglu_fused_ckpt(
    w0: torch.Tensor,  # [b, dh, dk]
    w1: torch.Tensor,  # [b, dv, dh]
    w2: torch.Tensor,  # [b, dh, dk]
    q: torch.Tensor,  # [b, l, dk]
    k: torch.Tensor,  # [b, l, dk]
    v: torch.Tensor,  # [b, l, dv]
    lr0: torch.Tensor,  # [b, l, 1]
    lr1: torch.Tensor,  # [b, l, 1]
    lr2: torch.Tensor,  # [b, l, 1]
    ttt_ua_order: list
) -> torch.Tensor:
    """
    Fused-kernel variant of the bidirectional operator. Requires the optional
    triton kernels (see `_FUSED_KERNELS_AVAILABLE`).

    Note this function takes flattened k, v and lr.
      by flattent, the batch dimension B is merged into the sequence dimension L.

    The query Q is not flattend.
    """

    BatchSize = q.size(0)
    # adding detach here sometimes improves stability.
    w0_norm = w0.detach().norm(dim=2, keepdim=True)
    w1_norm = w1.detach().norm(dim=2, keepdim=True)
    w2_norm = w2.detach().norm(dim=2, keepdim=True)

    output = []
    for start, end, update, apply in ttt_ua_order:
        # all tokens
        if end == -1:
            end = q.shape[1]

        ######### update the fast weight w0, w1, w2 with test-time training #########
        if update:
            ki, vi = k[:, start:end, :], v[:, start:end, :]
            lr0i = lr0[:, start:end, :]
            lr1i = lr1[:, start:end, :]
            lr2i = lr2[:, start:end, :]
            # make the shape to be (b, 1, l)
            lr0i, lr1i, lr2i = lr0i.reshape(BatchSize, 1, -1), lr1i.reshape(BatchSize, 1, -1), lr2i.reshape(BatchSize, 1, -1)
            # [BatchSize, Hidden, D] for dw0, dw2, [BatchSize, D, Hidden] for dw1
            dw0, dw1, dw2 = lact_swiglu_ffn_fast_weight_grads_with_ckpt(
                w0,
                w1,
                w2,
                ki,
                vi,
                lr0i,
                lr1i,
                lr2i,
            )


            dw0 = zeropower_via_newtonschulz5(dw0, 5)
            dw1 = zeropower_via_newtonschulz5(dw1, 5)
            dw2 = zeropower_via_newtonschulz5(dw2, 5)

            w1 = w1 + dw1
            w0 = w0 + dw0
            w2 = w2 + dw2

            w0 = w0 / (w0.norm(dim=2, keepdim=True) + 1e-5) * w0_norm
            w1 = w1 / (w1.norm(dim=2, keepdim=True) + 1e-5) * w1_norm
            w2 = w2 / (w2.norm(dim=2, keepdim=True) + 1e-5) * w2_norm

        ######### apply the updated fast weights to the query #########
        if apply:
            qi = q[:, start:end, :]
            oi = fused_swiglu_ffn_fwd_with_ckpt(w0, w1, w2, qi)
            output.append(oi)
    output = torch.cat(output, dim=1)

    return output, w0, w1, w2


class FastWeightGluMLPMultihead(nn.Module):
    """
    Multi-head SwiGLU fast-weight (TTT) layer.

    Set `causal=True` to use the shifted block-causal operator (for the vision
    expert, which generates image tokens autoregressively). Set `causal=False`
    (default) for the bidirectional operator (for the action expert, whose
    action chunk is denoised jointly).

    On init of fast_weight:

    Let's start with the magnitude of the value.
    value_proj is initialized with uniform distribution with range [-1.0/sqrt(d), 1.0/sqrt(d)]
        x is layernormed. So during init, value is unit norm total (not per head, per head is 1.0/sqrt(num_head))
        After silu, value is around norm of 2.7 per head.  (why? seems wired)

    Then for the fast weight, assume initial lr = 0.
    Then with l2_norm of q,k, input is unit normed.
    if w0 is initialized with kaiming, relu(w0 @ q) is unit normed.
    Then w1 is initialized with kaiming, so w1 @ relu(w0 @ q) is of norm sqrt(2) per head
    Since I compute total norm, it is sqrt(2) * sqrt(num_head), which is around 2.7 for dim=512, num_head=4.
    """

    def __init__(
        self,
        dim: int,
        head_dim: int,
        inter_multi: int = 1,
        bias: bool = False,
        base_lr=0.01,
        muon_update_steps=0,
        use_gate_fn = False,
        use_fused_kernels: bool = False,
        causal: bool = False,
        chunk_size: int = 64,
        vlm_hidden_size: int = None,
        use_cuda_kernel: bool = False,
    ):
        """
        Args:
            dim: input dimension, which should be the same as the local window attention dim and output dimension
            head_dim: dimension of each head
            inter_multi: the hidden dimension is head_dim * inter_multi
            bias: whether to use bias in linear layers
            base_lr: the base learning rate for the fast weight update
            muon_update_steps: number of steps for muon update
            use_gate_fn: whether to use gate function after the output
            causal: if True, use the shifted block-causal operator (vision expert);
                    if False, use the bidirectional operator (action expert)
            chunk_size: TTT chunk size for the causal operator
            vlm_hidden_size: if set (to any positive int), enables "method-B"
                context injection. The caller passes context tokens ALREADY
                projected to the mixer dim (`dim`) via `forward(..., ctx=...)`;
                these are fed as extra key/value into the fast-weight update
                (bidirectional) or as a global non-causal pre-update (causal),
                matching MoEBlock/MoEGeneratorBlock's `kv = cat([x, vlm_feat])`.
                The value itself is unused beyond enabling the lr head.
        """
        super().__init__()
        self.dim = dim
        assert dim % head_dim == 0
        self.num_heads = dim // head_dim
        self.head_dim = head_dim
        self.muon_update_steps = muon_update_steps
        self.causal = causal
        self.chunk_size = chunk_size
        self.vlm_hidden_size = vlm_hidden_size
        self.inject_ctx = vlm_hidden_size is not None
        # CUDA-accelerated causal forward (muon_update_steps==0 only). Falls back
        # to the torch op if the extension is unavailable or muon>0.
        self.use_cuda_kernel = use_cuda_kernel and causal and muon_update_steps == 0

        d_in = d_out = head_dim
        d_h = int(head_dim * inter_multi)

        gain = math.sqrt(2)  # for relu activations
        self.w0 = nn.Parameter(
            torch.randn(self.num_heads, d_in, d_h) * gain / math.sqrt(d_in)
        )  # [d_h * num_heads,  d_in]
        self.w1 = nn.Parameter(
            torch.randn(self.num_heads, d_h, d_out) * gain / math.sqrt(d_h)
        )  # [d_in * num_heads,  d_h]
        self.w2 = nn.Parameter(
            torch.randn(self.num_heads, d_in, d_h) * gain / math.sqrt(d_in)
        )  # [d_h * num_heads,  d_in]

        self.to_qkv = nn.Linear(dim, 3 * dim, bias=bias)
        self.c_proj = nn.Linear(dim, dim, bias=bias)

        self.lr_dim = self.num_heads
        self.lr_fc = nn.Linear(dim, self.lr_dim * 3)
        self.base_lr_inv = inv_softplus(base_lr)

        # Method-B context injection: caller passes context already projected to
        # the mixer dim; we only need a per-context-token learning-rate head so
        # the context participates in the fast-weight update like extra KV.
        if self.inject_ctx:
            self.ctx_lr_fc = nn.Linear(dim, self.lr_dim * 3)
        else:
            self.ctx_lr_fc = None

        self.use_gate_fn = use_gate_fn
        if self.use_gate_fn:
            self.gate_fn = nn.Sequential(
                nn.Linear(dim, dim, bias=bias),
                nn.SiLU()
            )
        self.use_fused_kernels = use_fused_kernels
        if self.use_fused_kernels and not _FUSED_KERNELS_AVAILABLE:
            raise ImportError(
                "use_fused_kernels=True but the fused triton kernels "
                "(lact_with_act_ckpt_plain) are not available in VLANeXt."
            )
        if self.use_fused_kernels and self.causal:
            raise NotImplementedError(
                "Fused kernels are only wired for the bidirectional operator."
            )
        self.o_norm = RMSNorm(head_dim, eps=1e-5, elementwise_affine=True)

    def forward(self, x: torch.Tensor, info={}, ctx: torch.Tensor = None, *args):
        """
        x: (b, l, d) -- expert tokens (query side).
        ctx: (b, t_ctx, d) -- optional context tokens ALREADY projected to the
            mixer dim `d` (method B). For the action/vision experts this is
            cat([vlm_proj(vlm_feat), gen_proj(gen_feat)]). Injected as extra KV
            into the fast-weight update.
        """
        B = x.shape[0]
        qkv = F.silu(self.to_qkv(x), inplace=True)  # Silu - Linear
        q, k, v = rearrange(
            qkv, "b l (qkv h d) -> qkv (b h) l d",
            qkv=3, h=self.num_heads
        )
        q = q / (q.norm(dim=2, keepdim=True) + 1e-5).to(x.dtype)
        k = k / (k.norm(dim=2, keepdim=True) + 1e-5).to(x.dtype)

        lr = self.lr_fc(x)  # [b, l, lr_dim]
        lr = torch.nn.functional.softplus(lr.float() + self.base_lr_inv)


        lr0, lr1, lr2 = rearrange(
            lr, "b l (lrs h d) -> lrs (b h) l d",
            lrs=3, h=self.num_heads
        )

        # ----- Method-B context injection: ctx already in mixer dim -> k/v tokens
        ctx_k = ctx_v = ctx_lr0 = ctx_lr1 = ctx_lr2 = None
        if ctx is not None:
            assert self.inject_ctx, (
                "ctx passed but vlm_hidden_size was not set at construction.")
            ctx_kv = F.silu(ctx, inplace=False)  # treat ctx like the silu(to_qkv) output
            ctx_k = rearrange(ctx_kv, "b t (h d) -> (b h) t d", h=self.num_heads)
            ctx_v = ctx_k
            ctx_k = ctx_k / (ctx_k.norm(dim=2, keepdim=True) + 1e-5).to(x.dtype)
            ctx_lr = self.ctx_lr_fc(ctx)
            ctx_lr = torch.nn.functional.softplus(ctx_lr.float() + self.base_lr_inv)
            ctx_lr0, ctx_lr1, ctx_lr2 = rearrange(
                ctx_lr, "b t (lrs h d) -> lrs (b h) t d",
                lrs=3, h=self.num_heads
            )

        if self.use_fused_kernels:
            if "w0" in info:
                assert "w1" in info and "w2" in info
                w0, w1, w2 = info["w0"], info["w1"], info["w2"]
            else:
                w0 = self.w0.transpose(-1, -2).repeat(B, 1, 1)
                w1 = self.w1.transpose(-1, -2).repeat(B, 1, 1)
                w2 = self.w2.transpose(-1, -2).repeat(B, 1, 1)

            output, w0, w1, w2 = bidirectional_lact_swiglu_fused_ckpt(
                w0, w1, w2, q, k, v, lr0, lr1, lr2, info["ttt_op_order"],
            )
        else:
            if "w0" in info:
                assert "w1" in info and "w2" in info
                w0, w1, w2 = info["w0"], info["w1"], info["w2"]
            else:
                w0 = self.w0.repeat(B, 1, 1)
                w1 = self.w1.repeat(B, 1, 1)
                w2 = self.w2.repeat(B, 1, 1)

            if self.causal:
                # Shifted block-causal (vision expert). Context, if any, is a
                # global non-causal pre-update so image->context is fully visible.
                if self.use_cuda_kernel:
                    from .ttt_cuda import causal_ttt
                    # causal_ttt routes to the CUDA forward when the extension is
                    # built + inputs are CUDA, else to the torch reference; its
                    # backward is exact (recompute) either way.
                    output, w0, w1, w2 = causal_ttt(
                        w0, w1, w2, q, k, v, lr0, lr1, lr2,
                        chunk_size=self.chunk_size,
                        vlm_k=ctx_k, vlm_v=ctx_v,
                        vlm_lr0=ctx_lr0, vlm_lr1=ctx_lr1, vlm_lr2=ctx_lr2,
                    )
                else:
                    output, w0, w1, w2 = causal_block_fast_weight_swish_glu(
                        w0, w1, w2, q, k, v, lr0, lr1, lr2,
                        chunk_size=self.chunk_size,
                        muon_update_steps=self.muon_update_steps,
                        vlm_k=ctx_k, vlm_v=ctx_v,
                        vlm_lr0=ctx_lr0, vlm_lr1=ctx_lr1, vlm_lr2=ctx_lr2,
                    )
            else:
                # Bidirectional (action expert). Context tokens are appended to
                # the update k/v sequence (apply queries stay expert-only),
                # matching MoEBlock's kv = cat([x, vlm_feat, gen_feat]).
                if ctx_k is not None:
                    L = q.shape[1]
                    upd_k = torch.cat([k, ctx_k], dim=1)
                    upd_v = torch.cat([v, ctx_v], dim=1)
                    upd_lr0 = torch.cat([lr0, ctx_lr0], dim=1)
                    upd_lr1 = torch.cat([lr1, ctx_lr1], dim=1)
                    upd_lr2 = torch.cat([lr2, ctx_lr2], dim=1)
                    # Update uses the full [expert + context] k/v sequence; apply
                    # uses only the expert queries (length L). The operator slices
                    # q and k independently per segment, so q stays length L.
                    ua_order = [
                        TTTOperator(start=0, end=upd_k.shape[1], update=True, apply=False),
                        TTTOperator(start=0, end=L, update=False, apply=True),
                    ]
                    output, w0, w1, w2 = fast_weight_swish_glu_weight_norm_mini_batch_apply(
                        w0, w1, w2, q, upd_k, upd_v, upd_lr0, upd_lr1, upd_lr2,
                        ua_order, muon_update_steps=self.muon_update_steps,
                    )
                else:
                    output, w0, w1, w2 = fast_weight_swish_glu_weight_norm_mini_batch_apply(
                        w0, w1, w2, q, k, v, lr0, lr1, lr2, info["ttt_op_order"],
                        muon_update_steps=self.muon_update_steps,
                    )


        if self.use_gate_fn:
            output = self.o_norm(output) * self.gate_fn(x)
        else:
            output = self.o_norm(output)

        output = rearrange(
            output, "(b h) l d -> b l (h d)", h=self.num_heads, b=B
        )

        output = self.c_proj(output)
        return output, {"w0": w0, "w1": w1, "w2": w2}

    @torch.no_grad()
    def infer_build_state(self, ctx: torch.Tensor):
        """Inference (causal, chunk>=seq) state = the VLM-pre-updated fast weights.

        At chunk_size >= seq_len the causal op is a SINGLE chunk: apply uses only
        the weights produced by the VLM global pre-update (image tokens never
        update before they are applied). So every image position is independent
        and depends only on (q_j, w_vlm). We precompute w_vlm ONCE per layer and
        reuse it for all autoregressive steps -> O(1) TTT work per step.

        ctx: (b, t_ctx, d) context already projected to mixer dim (== forward's ctx).
        Returns a state dict {w0,w1,w2} (the post-pre-update fast weights).
        """
        assert self.causal and self.inject_ctx, "incremental path is for causal method-B"
        B = ctx.shape[0]
        w0 = self.w0.repeat(B, 1, 1)
        w1 = self.w1.repeat(B, 1, 1)
        w2 = self.w2.repeat(B, 1, 1)
        # build ctx k/v/lr exactly as forward() does
        ctx_kv = F.silu(ctx, inplace=False)
        ctx_k = rearrange(ctx_kv, "b t (h d) -> (b h) t d", h=self.num_heads)
        ctx_v = ctx_k
        ctx_k = ctx_k / (ctx_k.norm(dim=2, keepdim=True) + 1e-5).to(ctx.dtype)
        ctx_lr = self.ctx_lr_fc(ctx)
        ctx_lr = torch.nn.functional.softplus(ctx_lr.float() + self.base_lr_inv)
        ctx_lr0, ctx_lr1, ctx_lr2 = rearrange(
            ctx_lr, "b t (lrs h d) -> lrs (b h) t d", lrs=3, h=self.num_heads)
        # one global (non-causal) pre-update == the vlm_k branch of the causal op
        w0, w1, w2 = _vlm_preupdate(w0, w1, w2, ctx_k, ctx_v, ctx_lr0, ctx_lr1, ctx_lr2)
        return {"w0": w0, "w1": w1, "w2": w2}

    @torch.no_grad()
    def infer_step(self, x_new: torch.Tensor, state: dict):
        """Apply cached fast weights to the NEW token(s) only. O(1) in seq len.

        x_new: (b, n_new, d) -- typically n_new == 1 (the latest token).
        state: from infer_build_state. Returns output (b, n_new, d).
        Numerically identical to forward()'s apply for those positions.
        """
        # CUDA-Graph fast path: capture the whole single-token step once, then
        # replay -> eliminates per-call python dispatch + launch overhead.
        # DISABLED by default: graph capture fixes buffer addresses, which under
        # the dynamic per-step allocation of a real eval rollout leads to
        # "illegal memory access" / CUBLAS_STATUS_EXECUTION_FAILED after many
        # replays (verified 2026-06-19: incremental eval crashed every episode ->
        # caught by the eval try/except -> silent SR=0). The eager path keeps the
        # fused mid-kernel (still O(n)); only the per-call launch overhead returns.
        # Opt back in by setting blk.attn._use_infer_graph=True if buffers are static.
        if (self.use_cuda_kernel and x_new.shape[1] == 1 and not self.use_gate_fn
                and x_new.is_cuda and getattr(self, "_use_infer_graph", False)):
            out = self._infer_step_graphed(x_new, state)
            if out is not None:
                return out
        return self._infer_step_eager(x_new, state)

    @torch.no_grad()
    def _infer_step_graphed(self, x_new, state):
        """CUDA-graph-captured single token step. Returns None if capture is not
        possible (falls back to eager). Keyed by (state tensors identity, dtype)."""
        from .ttt_cuda import _load_extension
        ext = _load_extension()
        if ext is None or not hasattr(ext, "infer_step_mid"):
            return None
        key = (state["w0"].data_ptr(), x_new.dtype, x_new.shape[0])
        cache = getattr(self, "_graph_cache", None)
        if cache is None:
            cache = self._graph_cache = {}
        g = cache.get(key)
        if g is None:
            g = self._build_infer_graph(x_new, state, ext)
            if g is None:
                self._use_infer_graph = False  # capture failed once -> stop trying
                return None
            cache[key] = g
        g["static_in"].copy_(x_new)
        g["graph"].replay()
        # clone: static_out is reused every replay; callers that collect outputs
        # across steps would otherwise all alias the same buffer.
        return g["static_out"].clone()

    @torch.no_grad()
    def _build_infer_graph(self, x_new, state, ext):
        try:
            static_in = x_new.clone()
            w0, w1, w2 = state["w0"], state["w1"], state["w2"]
            onw = self.o_norm.weight.to(x_new.dtype)
            Wq = self.to_qkv.weight[:self.dim]
            bq = self.to_qkv.bias[:self.dim] if self.to_qkv.bias is not None else None

            def _run(x):
                q_lin = F.silu(F.linear(x, Wq, bq), inplace=False)
                q_un = rearrange(q_lin, "b l (h d) -> (b h) (l d)", h=self.num_heads)
                o = ext.infer_step_mid(q_un, w0, w2, w1, onw, 1e-5, 1e-5)
                out = rearrange(o.unsqueeze(1), "(b h) l d -> b l (h d)",
                                h=self.num_heads, b=x.shape[0])
                return self.c_proj(out)

            # warmup on a side stream (required before capture)
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    _run(static_in)
            torch.cuda.current_stream().wait_stream(s)

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                static_out = _run(static_in)
            return {"graph": graph, "static_in": static_in, "static_out": static_out}
        except Exception:
            return None

    @torch.no_grad()
    def _infer_step_eager(self, x_new: torch.Tensor, state: dict):
        B = x_new.shape[0]
        w0, w1, w2 = state["w0"], state["w1"], state["w2"]
        # Incremental apply only needs q (k,v feed the update, which is dropped at
        # chunk>=seq). to_qkv is Linear(dim, 3*dim); compute ONLY the q rows
        # (first `dim` outputs) -> 1/3 the GEMM, bit-identical to slicing the full.
        Wq = self.to_qkv.weight[:self.dim]
        bq = self.to_qkv.bias[:self.dim] if self.to_qkv.bias is not None else None
        q_lin = F.silu(F.linear(x_new, Wq, bq), inplace=False)
        if self.use_cuda_kernel and x_new.shape[1] == 1 and not self.use_gate_fn:
            from .ttt_cuda import _load_extension
            ext = _load_extension()
            if ext is not None and hasattr(ext, "infer_step_mid"):
                # Stage-1 mega: q-norm + apply + RMSNorm in ONE kernel.
                # Pass UN-normalized per-head q; the kernel does the L2 q-norm.
                q_un = rearrange(q_lin, "b l (h d) -> (b h) (l d)", h=self.num_heads)
                onw = self.o_norm.weight.to(q_un.dtype)
                o = ext.infer_step_mid(q_un, w0, w2, w1, onw, 1e-5, 1e-5)  # [B*h, dh]
                out = rearrange(o.unsqueeze(1), "(b h) l d -> b l (h d)",
                                h=self.num_heads, b=B)
                return self.c_proj(out)
        q = rearrange(q_lin, "b l (h d) -> (b h) l d", h=self.num_heads)
        q = q / (q.norm(dim=2, keepdim=True) + 1e-5).to(x_new.dtype)
        if self.use_cuda_kernel and x_new.shape[1] == 1:
            from .ttt_cuda import _load_extension
            ext = _load_extension()
            if ext is not None and hasattr(ext, "infer_step"):
                # fused apply + RMSNorm: q is [B*h, 1, dh] -> squeeze the token dim
                qn = q.squeeze(1).contiguous()       # [B*h, dh]
                onw = self.o_norm.weight.to(qn.dtype)
                o = ext.infer_step(qn, w0, w2, w1, onw, 1e-5)  # [B*h, dh]
                out = o.unsqueeze(1)
                if self.use_gate_fn:
                    out = out * self.gate_fn(x_new).reshape(
                        B * self.num_heads, 1, self.head_dim)
                out = rearrange(out, "(b h) l d -> b l (h d)", h=self.num_heads, b=B)
                return self.c_proj(out)
        out = (F.silu(q @ w0, inplace=False) * (q @ w2)) @ w1
        out = self.o_norm(out)
        if self.use_gate_fn:
            out = out * self.gate_fn(x_new)  # note: gate uses x_new (per-position, ok)
        out = rearrange(out, "(b h) l d -> b l (h d)", h=self.num_heads, b=B)
        return self.c_proj(out)

    def extra_repr(self) -> str:
        return (f"w0 shape: {self.w0.shape}, w1 shape: {self.w1.shape}, w2 shape: {self.w2.shape}, "
                f"causal: {self.causal}, chunk_size: {self.chunk_size}, "
                f"Muon update steps: {self.muon_update_steps}, "
                f"Base lr: {math.log(1 + math.exp(self.base_lr_inv))}, ")
