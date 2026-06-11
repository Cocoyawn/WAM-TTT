import torch
import torch.nn as nn

from .ttt import FastWeightGluMLPMultihead
from .linear_attn_mixer import LinearAttnMixer


def _mixer_at(layer_idx, mixer_type, mix_every_n, fallback_mixer="attention"):
    """[F,F,F,M] interleave: every mix_every_n-th layer (1-based) uses the main
    mixer `mixer_type`; the rest use `fallback_mixer` (default 'attention', i.e. the
    classic [A,A,A,M] pattern). Set fallback_mixer='swa' to reproduce Kairos's
    [SWA,SWA,SWA,GDN] stack where the non-linear-attn layers are sliding-window
    attention rather than full attention.
    mixer_type='attention' -> all-attention baseline (fallback is irrelevant)."""
    if mixer_type == "attention" or mix_every_n is None or mix_every_n <= 0:
        return "attention"
    return mixer_type if ((layer_idx + 1) % mix_every_n == 0) else fallback_mixer


def build_swa_causal_mask(T_img, T_ctx, window_size, device, dtype):
    """Sliding-window CAUSAL mask for the vision DiT attention, method-B layout.

    Query layout: T_img image tokens (causal, autoregressive). Key layout:
    [T_img image tokens ; T_ctx VLM-context tokens]. Returns an additive mask of
    shape (T_img, T_img + T_ctx) with 0 where allowed and -inf where masked.

    Rules (matching Kairos SWA semantics, adapted to our single-frame token grid):
      - image->image: query i attends to keys [max(0, i-window_size+1) .. i]
        (causal AND windowed; can't see the future, can't see beyond the window).
      - image->VLM ctx: ALWAYS visible (global context, never windowed) — preserves
        the method-B injection semantics identical to the full-attention baseline.

    window_size is in TOKENS (not frames). window_size>=T_img degrades to plain
    causal attention (full history visible), i.e. our existing baseline.
    """
    mask = torch.zeros((T_img, T_img + T_ctx), device=device, dtype=dtype)
    idx = torch.arange(T_img, device=device)
    # disallow future: key_pos > query_pos
    future = idx[None, :] > idx[:, None]
    # disallow too-old: key_pos < query_pos - (window_size - 1)
    too_old = idx[None, :] < (idx[:, None] - (window_size - 1))
    blocked = future | too_old                       # (T_img, T_img)
    mask[:, :T_img].masked_fill_(blocked, float("-inf"))
    # ctx columns (T_img:) stay 0 -> fully visible
    return mask


class MoEGeneratorBlock(nn.Module):
    def __init__(self, hidden_size, vlm_hidden_size, num_heads, mlp_ratio=4.0,
                 mixer_type="attention", ttt_chunk_size=16, layer_idx=0, swa_window_size=64):
        super().__init__()
        self.mixer_type = mixer_type
        self.swa_window_size = swa_window_size
        self.norm1 = nn.LayerNorm(hidden_size)
        if mixer_type in ("attention", "swa"):
            # 'swa' reuses the same MultiheadAttention but applies a sliding-window
            # causal mask in forward(); 'attention' uses the plain causal mask.
            self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        elif mixer_type == "ttt":
            # Method-B causal TTT mixer: image->image causal + image->VLM global.
            self.attn = FastWeightGluMLPMultihead(
                dim=hidden_size, head_dim=hidden_size // num_heads,
                causal=True, chunk_size=ttt_chunk_size,
                vlm_hidden_size=hidden_size,  # ctx already projected to hidden_size
            )
        elif mixer_type in ("gla", "gdn"):
            # Method-B causal linear-attention mixer: VLM ctx prepended so every
            # image token reads it; image->image stays causal (left-to-right).
            self.attn = LinearAttnMixer(
                kind=mixer_type, dim=hidden_size, num_heads=num_heads, layer_idx=layer_idx,
            )
        else:
            raise ValueError(f"Unknown mixer_type: {mixer_type}")
        self.vlm_proj = nn.Linear(vlm_hidden_size, hidden_size)

        self.norm2 = nn.LayerNorm(hidden_size)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size),
        )

    def forward(self, x, vlm_feat):
        x_norm = self.norm1(x)
        v_feat = self.vlm_proj(vlm_feat)

        if self.mixer_type in ("attention", "swa"):
            T_img = x.shape[1]
            T_vlm = vlm_feat.shape[1]
            if self.mixer_type == "swa":
                # sliding-window causal mask (image->image windowed+causal, image->VLM global)
                full_mask = build_swa_causal_mask(T_img, T_vlm, self.swa_window_size, x.device, x.dtype)
            else:
                full_mask = torch.zeros((T_img, T_img + T_vlm), device=x.device, dtype=x.dtype)
                causal_mask = torch.triu(torch.ones((T_img, T_img), device=x.device, dtype=torch.bool), diagonal=1)
                full_mask[:, :T_img].masked_fill_(causal_mask, float('-inf'))
            kv = torch.cat([x_norm, v_feat], dim=1)
            attn_out, _ = self.attn(query=x_norm, key=kv, value=kv, attn_mask=full_mask)
        else:  # ttt / gla / gdn: same causal+global-VLM semantics, method-B
            attn_out, _ = self.attn(x_norm, {}, v_feat)

        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x

class ImageGeneratorTransformer(nn.Module):
    """
    Autoregressive Transformer for Image Generation using MoE-like Layer-wise Cross Attention
    """
    def __init__(self, vocab_size, vlm_hidden_size, hidden_size=768, depth=12, num_heads=12, mlp_ratio=4.0, max_seq_len=1024,
                 mixer_type="attention", mix_every_n=4, ttt_chunk_size=16, fallback_mixer="attention", swa_window_size=64):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, hidden_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, hidden_size))

        self.blocks = nn.ModuleList([
            MoEGeneratorBlock(hidden_size, vlm_hidden_size, num_heads, mlp_ratio=mlp_ratio,
                              mixer_type=_mixer_at(i, mixer_type, mix_every_n, fallback_mixer),
                              ttt_chunk_size=ttt_chunk_size, layer_idx=i, swa_window_size=swa_window_size)
            for i in range(depth)
        ])

        self.norm_final = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size)
        
        self.initialize_weights()

    def initialize_weights(self):
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.token_emb.weight, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, input_ids, vlm_hidden_states):
        x = self.token_emb(input_ids)
        x = x + self.pos_embed[:, :x.shape[1], :]
        
        relevant_vlm_states = vlm_hidden_states[-len(self.blocks):]
        
        hidden_states = []
        for block, vlm_state in zip(self.blocks, relevant_vlm_states):
            vlm_state = vlm_state.to(dtype=x.dtype)
            x = block(x, vlm_state)
            hidden_states.append(x)
            
        x = self.norm_final(x)
        logits = self.head(x)
        
        return logits, hidden_states
