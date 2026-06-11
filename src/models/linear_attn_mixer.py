"""
Linear-attention token mixers (GatedDeltaNet / GLA) as drop-in replacements for
the inner attention of MoEBlock / MoEGeneratorBlock, using **method-B context
injection** to stay semantically identical to the softmax-attention baseline.

Method-B injection
------------------
The baseline MoEBlock attention does:  kv = cat([x, vlm_ctx]); attn(q=x, k=kv, v=kv)
i.e. each token attends to itself+others AND to the projected VLM/gen context.

GatedDeltaNet / GLA are pure self-mixing recurrences over an input sequence (q,k,v
all come from the same hidden_states; there is no external-KV port). To reproduce
"tokens can read the VLM context", we **prepend** the projected context tokens to the
input sequence, run the linear-attention recurrence over [ctx ; x], then **slice off**
the context prefix and keep only the outputs at the x positions:

    seq = cat([ctx, x], dim=1)            # ctx is already projected to hidden_size
    out_seq = linear_attn(seq)            # causal recurrence: x positions see ctx (which precedes them)
    out = out_seq[:, ctx_len:, :]         # keep only the x outputs

Because these operators are causal, putting ctx FIRST guarantees every x token can
read the full context (ctx precedes all x), exactly matching the attention mask where
x->ctx is fully visible. The x->x interaction is causal (left-to-right), which is the
intended behavior for the vision expert (autoregressive image tokens) and is exactly
the "action causality" we want to probe on the action expert.

This keeps the MoEBlock shell (adaLN / residual / MLP / vlm_proj / gen_proj) untouched
and only swaps the token-mixing operator, so an ablation attention-vs-TTT-vs-GLA-vs-GDN
differs ONLY in the operator.
"""
import torch
import torch.nn as nn

from .fla.layers.gated_deltanet import GatedDeltaNet


class LinearAttnMixer(nn.Module):
    """Wraps a fla linear-attention layer (GatedDeltaNet / GLA) with method-B
    context injection and the MoEBlock calling convention `forward(x, info, ctx)
    -> (out, aux)`.

    Args:
        kind: "gdn" (GatedDeltaNet) or "gla" (GatedLinearAttention, if vendored).
        dim: model hidden size (= MoEBlock hidden_size).
        num_heads: number of heads for the linear-attention layer.
        head_dim: per-head dim (GatedDeltaNet needs num_heads*head_dim ~= 0.75*dim
                  by its own design; we pass dim//num_heads by default and let the
                  layer's own asserts guard validity).
    """

    def __init__(self, kind, dim, num_heads, head_dim=None, layer_idx=0):
        super().__init__()
        self.kind = kind
        self.dim = dim
        if kind == "gdn":
            # GatedDeltaNet's parameter budget assumes num_heads*head_dim = 0.75*dim.
            # Use head_dim = (0.75*dim)//num_heads when not given, rounded to a
            # multiple that keeps key_dim divisible by num_heads.
            if head_dim is None:
                hd = int(0.75 * dim) // num_heads
                hd = max(16, (hd // 16) * 16)  # keep a sane, kernel-friendly head_dim
            else:
                hd = head_dim
            self.layer = GatedDeltaNet(
                hidden_size=dim,
                head_dim=hd,
                num_heads=num_heads,
                expand_v=2,
                mode="chunk",
                use_gate=True,
                use_short_conv=True,
                layer_idx=layer_idx,
            )
        elif kind == "gla":
            from .fla.layers.gla import GatedLinearAttention  # vendored separately
            self.layer = GatedLinearAttention(
                hidden_size=dim,
                num_heads=num_heads,
                mode="chunk",
                use_short_conv=False,
                layer_idx=layer_idx,
            )
        else:
            raise ValueError(f"Unknown linear-attn kind: {kind}")

    def forward(self, x, info=None, ctx=None, *args):
        """x: [B, L, dim]; ctx: [B, M, dim] already projected (method-B). Returns
        (out[B, L, dim], None) to match the (attn_out, _) unpacking in MoEBlock."""
        if ctx is not None and ctx.shape[1] > 0:
            ctx_len = ctx.shape[1]
            seq = torch.cat([ctx, x], dim=1)
        else:
            ctx_len = 0
            seq = x
        out_seq, _, _ = self.layer(seq)
        out = out_seq[:, ctx_len:, :]
        return out, None
