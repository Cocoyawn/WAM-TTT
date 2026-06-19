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
                 mixer_type="attention", ttt_chunk_size=16, layer_idx=0, swa_window_size=64,
                 ttt_use_cuda_kernel=False):
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
                use_cuda_kernel=ttt_use_cuda_kernel,
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

    # ------------------------------------------------------------------
    # Incremental (O(1)-per-token) decode. Numerically mirrors forward()'s
    # apply for the new position; used by ImageGeneratorTransformer.generate_incremental.
    # Only 'ttt' and plain 'attention' blocks are supported (the deployed
    # [A,A,A,T] stack); 'swa'/'gla'/'gdn' fall back to None (caller recomputes).
    # ------------------------------------------------------------------
    @torch.no_grad()
    def infer_init(self, vlm_feat):
        """Build the per-block decode state once. vlm_feat: (B, T_vlm, vlm_hidden)."""
        if self.mixer_type == "ttt":
            ctx = self.vlm_proj(vlm_feat.to(self.vlm_proj.weight.dtype))
            return {"kind": "ttt", "ttt": self.attn.infer_build_state(ctx)}
        if self.mixer_type == "attention":
            # cache the VLM key/value (constant across steps) + grow image K/V.
            E = self.attn.embed_dim
            ipw, ipb = self.attn.in_proj_weight, self.attn.in_proj_bias
            v_feat = self.vlm_proj(vlm_feat.to(self.vlm_proj.weight.dtype))  # (B,T_vlm,E)
            Wk, Wv = ipw[E:2 * E], ipw[2 * E:]
            bk = ipb[E:2 * E] if ipb is not None else None
            bv = ipb[2 * E:] if ipb is not None else None
            k_vlm = torch.nn.functional.linear(v_feat, Wk, bk)
            v_vlm = torch.nn.functional.linear(v_feat, Wv, bv)
            return {"kind": "attention", "k_vlm": k_vlm, "v_vlm": v_vlm,
                    "k_img": None, "v_img": None}
        return None  # unsupported mixer for incremental decode

    @torch.no_grad()
    def infer_step(self, x, state):
        """One token through the whole block (norm1 -> mixer -> residual ->
        norm2 -> mlp). x: (B, 1, hidden). Returns updated x (B, 1, hidden)."""
        x_norm = self.norm1(x)
        if state["kind"] == "ttt":
            attn_out = self.attn.infer_step(x_norm, state["ttt"])
        else:  # attention: KV-cached single-token attention == full causal+global-VLM
            attn = self.attn
            E, H = attn.embed_dim, attn.num_heads
            dh = E // H
            B = x_norm.shape[0]
            ipw, ipb = attn.in_proj_weight, attn.in_proj_bias
            Wq, Wk, Wv = ipw[:E], ipw[E:2 * E], ipw[2 * E:]
            bq = ipb[:E] if ipb is not None else None
            bk = ipb[E:2 * E] if ipb is not None else None
            bv = ipb[2 * E:] if ipb is not None else None
            q = torch.nn.functional.linear(x_norm, Wq, bq)  # (B,1,E)
            k = torch.nn.functional.linear(x_norm, Wk, bk)
            v = torch.nn.functional.linear(x_norm, Wv, bv)
            # grow image K/V cache (causal: token t sees image 0..t)
            state["k_img"] = k if state["k_img"] is None else torch.cat([state["k_img"], k], dim=1)
            state["v_img"] = v if state["v_img"] is None else torch.cat([state["v_img"], v], dim=1)
            K = torch.cat([state["k_img"], state["k_vlm"]], dim=1)  # (B,Lk,E)
            V = torch.cat([state["v_img"], state["v_vlm"]], dim=1)
            Lk = K.shape[1]
            qh = q.view(B, 1, H, dh).transpose(1, 2)       # (B,H,1,dh)
            Kh = K.view(B, Lk, H, dh).transpose(1, 2)
            Vh = V.view(B, Lk, H, dh).transpose(1, 2)
            scaling = dh ** -0.5
            attn_w = torch.softmax((qh * scaling) @ Kh.transpose(-2, -1), dim=-1)
            o = (attn_w @ Vh).transpose(1, 2).reshape(B, 1, E)  # (B,1,E)
            attn_out = attn.out_proj(o)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x



class ImageGeneratorTransformer(nn.Module):
    """
    Autoregressive Transformer for Image Generation using MoE-like Layer-wise Cross Attention
    """
    def __init__(self, vocab_size, vlm_hidden_size, hidden_size=768, depth=12, num_heads=12, mlp_ratio=4.0, max_seq_len=1024,
                 mixer_type="attention", mix_every_n=4, ttt_chunk_size=16, fallback_mixer="attention", swa_window_size=64,
                 ttt_use_cuda_kernel=False):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, hidden_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, hidden_size))

        self.blocks = nn.ModuleList([
            MoEGeneratorBlock(hidden_size, vlm_hidden_size, num_heads, mlp_ratio=mlp_ratio,
                              mixer_type=_mixer_at(i, mixer_type, mix_every_n, fallback_mixer),
                              ttt_chunk_size=ttt_chunk_size, layer_idx=i, swa_window_size=swa_window_size,
                              ttt_use_cuda_kernel=ttt_use_cuda_kernel)
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

    @torch.no_grad()
    def generate_incremental(self, vlm_hidden_states, num_tokens):
        """O(n) autoregressive decode: replaces predict_action's O(n^2) loop
        (full generator.forward on the growing prefix at every step). Each TTT
        block caches its VLM-pre-updated fast weights once and applies infer_step
        per token; each attention block keeps a KV-cache. Positions are
        independent at chunk>=seq for TTT, and KV-cache is exact for attention,
        so the produced token IDs and the per-layer hidden_states match the
        full-recompute path within round-off.

        Returns (token_ids (B, num_tokens), hidden_states list[(B, num_tokens, H)]),
        matching predict_action's `curr_ids[:, 1:]` and the line-887 full forward.
        Returns None if any block uses an unsupported mixer (caller recomputes)."""
        blocks = self.blocks
        relevant_vlm = vlm_hidden_states[-len(blocks):]
        dtype = next(self.parameters()).dtype
        states = []
        for blk, vfeat in zip(blocks, relevant_vlm):
            st = blk.infer_init(vfeat.to(dtype=dtype))
            if st is None:
                return None  # unsupported mixer -> let caller fall back
            states.append(st)

        B = vlm_hidden_states[0].shape[0]
        device = vlm_hidden_states[0].device
        curr_id = torch.zeros((B, 1), dtype=torch.long, device=device)  # BOS-like start
        out_ids = []
        hs_per_layer = [[] for _ in blocks]
        for t in range(num_tokens):
            x = self.token_emb(curr_id) + self.pos_embed[:, t:t + 1, :]
            for li, (blk, st) in enumerate(zip(blocks, states)):
                x = blk.infer_step(x, st)
                hs_per_layer[li].append(x)
            logits = self.head(self.norm_final(x))      # (B,1,V)
            nxt = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            out_ids.append(nxt)
            curr_id = nxt
        token_ids = torch.cat(out_ids, dim=1)           # (B, num_tokens)
        hidden_states = [torch.cat(h, dim=1) for h in hs_per_layer]
        return token_ids, hidden_states
