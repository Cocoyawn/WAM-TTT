# 当前 TTT Layer 架构图（FastWeightGluMLPMultihead，方式 B）

源码：`src/models/ttt.py::FastWeightGluMLPMultihead.forward`
作为 token-mixer 嵌在 MoEBlock(action)/ MoEGeneratorBlock(vision) 的外壳里，
只替换内部 attention 那一次调用，adaLN / 残差 / MLP / norm 全部保留。

```
                            输入
        x: [B, L, D]  (expert tokens, query 侧)          ctx: [B, T, D]  (VLM/gen feat，已投影到 D，方式 B)
        L = 8 (action) / 256 (vision)                    T = VLM token 数 (+gen)
              │                                                  │
              ▼                                                  ▼
   ┌──────────────────────┐                          ┌──────────────────────┐
   │ to_qkv: Linear(D,3D)  │                          │  ctx 直接当 silu 输出  │
   │ + SiLU                │                          │  silu(ctx)            │
   └──────────┬───────────┘                          └──────────┬───────────┘
              │ split q,k,v  (按 head 拆: num_heads)             │ ctx_k = ctx_v = silu(ctx)
              ▼                                                  ▼
   q,k,v: [(B·h), L, hd]                              ctx_k,ctx_v: [(B·h), T, hd]
              │                                                  │
   ┌──────────┴──────────┐                            ┌──────────┴──────────┐
   │ L2 norm: q,k 单位化  │                            │ L2 norm: ctx_k 单位化 │
   └──────────┬──────────┘                            └──────────┬──────────┘
              │                                                  │
   ┌──────────┴──────────┐                            ┌──────────┴──────────┐
   │ lr_fc: Linear(D,3h)  │  每 token 学习率           │ ctx_lr_fc:Linear(D,3h)│ 每 ctx-token 学习率
   │ softplus(·+base_lr)  │  → lr0,lr1,lr2            │ softplus → ctx_lr0/1/2 │
   └──────────┬──────────┘                            └──────────┬──────────┘
              │                                                  │
              └──────────────────────┬───────────────────────────┘
                                     ▼
              ┌────────────────────────────────────────────────┐
              │            快权重 (fast weights)                  │
              │  从慢权重复制: w0,w1,w2 = self.w{0,1,2}.repeat(B) │
              │  慢权重 nn.Parameter (Kaiming), 存 checkpoint     │
              │  快权重每次 forward 新建, test-time 训练, 不保存   │
              │  SwiGLU MLP:  f(z) = (silu(z@w0) · (z@w2)) @ w1  │
              └───────────────────────┬────────────────────────┘
                                      ▼
        ┌─────────────────────────────────────────────────────────────┐
        │                  分两种情况 (causal 标志)                       │
        │                                                               │
        │  ── 双向 (action expert, causal=False) ──                      │
        │   fast_weight_swish_glu_weight_norm_mini_batch_apply           │
        │   UPDATE 段: k/v = cat([k, ctx_k]), lr = cat([lr, ctx_lr])     │
        │              用全序列 [expert+ctx] 训练快权重 (8 tokens 双向)   │
        │   APPLY  段: 只用 expert query q (长度 L) 读快权重             │
        │   ⇒ 等价 attention 的 kv = cat([x, vlm_feat, gen_feat])        │
        │                                                               │
        │  ── 因果 (vision expert, causal=True) ──                       │
        │   causal_block_fast_weight_swish_glu (chunk_size 分块)         │
        │   ① 全局预更新: 先用 ctx_k/ctx_v 非因果更新快权重               │
        │      ⇒ 每个 image token 都能读 VLM (image→VLM 全可见)          │
        │   ② 分块循环 (256/chunk_size 块, apply-then-update):           │
        │      for 每块: 先 APPLY(用<本块的快权重) 再 UPDATE(本块k/v)     │
        │      ⇒ image→image 因果; 每块含 Newton-Schulz (Muon) 正交化    │
        └───────────────────────────────┬───────────────────────────────┘
                                        ▼
                          output: [(B·h), L, hd]
                                        │
                   ┌────────────────────┴────────────────────┐
                   │ o_norm: RMSNorm(head_dim)                │
                   │ (可选 gate_fn: ·SiLU(Linear(x)))          │
                   └────────────────────┬─────────────────────┘
                                        │ rearrange (b h) l d → b l (h d)
                                        ▼
                          ┌──────────────────────────┐
                          │ c_proj: Linear(D, D)      │
                          └────────────┬─────────────┘
                                       ▼
                  返回 (output[B,L,D],  {w0,w1,w2})   ← 训练好的快权重(不保存)
```

## 关键点
- **慢权重 vs 快权重**：慢权重 `w0/w1/w2`(nn.Parameter, Kaiming 初始化)存进 checkpoint；
  每次 forward 复制成快权重 `.repeat(B,1,1)`，在序列上做 test-time 训练，**不保存**(VLA 用法里无状态)。
- **SwiGLU 快权重 MLP**：`f(z) = (silu(z@w0) · (z@w2)) @ w1`，三个矩阵就是被 test-time 更新的对象。
- **方式 B 注入**：ctx (= vlm_proj/gen_proj 输出) 当额外 KV 参与快权重 UPDATE，但不参与 APPLY 的 query
  → 与被替换的 attention `kv=cat([x,vlm_feat])` 语义一致。
- **方向性**：action 双向(8 token 联合)；vision 因果(256 token 自回归，chunk 分块 apply-then-update，
  VLM 走全局预更新保持可见)。
- **更新算子**：Newton-Schulz 5 步正交化(Muon 风格)+ weight-norm，纯 PyTorch + @torch.compile，无手写 kernel。
- **chunk_size**(仅因果有效)：块数 = 256/chunk_size；越大越快(更新次数越少)，推理延迟随之下降。
