# VLANeXt 近两天实验计划(2026-06-07 起)

> 目标:在 chunk-size 扫描完成的基础上,引入 **Gated Linear Attention (GLA)** 与 **因果性消融**,
> 为"是否把 action DiT 换成 Gated DeltaNet 这类因果结构"提供判据。
> 参考实现:KairosDiT(`kairos-sensenova/kairos/modules/dits/kairos_dit.py`),它用
> `(i+1)%4==0` 的 [A,A,A,Linear] 模式把 GatedDeltaNet 当 token mixer —— **与我们 VLANeXt 的
> [A,A,A,T] 思想完全一致**,可直接借鉴。

---

## 🔴 高优先级(2026-06-10 新增)— TTT-256 跨 suite 训练

> **背景**:chunk-size 扫描确认 vision `chunk_size=256`(1 block/frame)在 LIBERO-spatial 上
> SR 仅比 chunk16 低 0.4pp(97.4% vs 97.8%),但部署推理快 2.3×,已成为标准默认(见 memory
> `vlanext-compute-resource-limits` / `ttt-chunk-size-train-speed`)。当前 TTT-256 只在 **spatial**
> 一个 suite 训练 + env-gen 评测(进行中)。要确立 TTT-256 是通用默认,需补齐另外 3 个 suite。

**目标**:在 LIBERO 其余 3 个 suite 上训练 TTT-256(vision chunk_size=256,world-model 模式),
与已有的 spatial TTT-256 一起构成完整 4-suite 成功率表,并和 attention/TTT-16 对照。

**前置条件(已核实 2026-06-10)**:
- 训练数据就绪:`~/data/LIBERO_modified/{libero_object_no_noops, libero_goal_no_noops, libero_10_no_noops}`。
- 模板:`config/ablation_wm_ttt_chunk256.yaml`(spatial 版),只需改 `task_suite_name` + `project.name` +
  `output_dir` 子目录。`generator_ttt_chunk_size: 256` / `policy_mixer_type/generator_mixer_type: ttt` 保持。
- 步数:config 注释 "10k for spatial/object/goal; 12k for long(10)";但当前 WM 模式跑的是 30k。
  **决策点(待 review)**:object/goal 用 30k 对齐 spatial,long(libero_10)用 30k 还是更多(任务更长)。
- save_interval 必须设 **10000**(默认 2000 会塞满盘,见资源 memory)。
- **标准规则**:2-GPU DDP(torchrun --nproc_per_node=2),~6h/run。PYTHONPATH=$REPO 否则 ModuleNotFoundError。

**子任务**:
- HP-1 train TTT-256 @ **libero_object**(30k, warmup 1500, WM)→ SR
- HP-2 train TTT-256 @ **libero_goal**(30k, warmup 1500, WM)→ SR
- HP-3 train TTT-256 @ **libero_10 / long**(**36k=30k×1.2, warmup 1800=1500×1.2**, WM)→ SR
- HP-4 4 个 suite 各自 base eval;汇总 spatial/object/goal/long × TTT-256 成功率表
- ~~HP-5 各 suite env-gen~~ **暂不做**(2026-06-10 决策:优先环境泛化主线)

**决策已定(2026-06-10)**:步数等比例放大,object/goal=30k,long=30k×1.2=36k,warmup 同比 1500→1800
(沿用 final_libero 历史 1.2× 先例 wu4500→wu5400)。**CONFIG 坑**:train.py 读 `data.max_steps`(非 train.max_steps)
+ `train.warmup_steps`,两参数在不同 section。configs 已生成核实:ablation_wm_ttt_chunk256_libero_{object,goal,long}.yaml。

**排程**:scripts/autostart_hp_trainings.sh 已后台运行——轮询每 5min,≥2 卡<40GB 且 RAM≥100GB 余量时自动启动
一个 2-GPU DDP 训练,3 个排队(object→goal→long)。当前等待中(RAM 不足)。TTT-256 env-gen(~5h)/GDN(~6h)腾资源后自动拉起。

---

## ✅ 已锁定决策(2026-06-07)

- **VLM 注入一律用方式 B**:把 VLM hidden state 投影后**拼进 token-mixing 算子的 KV/输入序列**,
  而非 KairosDiT 的独立 cross-attn。TTT / GLA / GatedDeltaNet 三者**只换 token-mixing 算子**,
  其余(adaLN/残差/MLP/VLM 注入方式)全同 → 消融最可比。这是整个实验族的统一前提。

---

## 阶段 0(进行中)— 当前正在跑的实验,属于本计划的一部分

这三组实验是本计划的"地基",chunk 扫描结果直接喂给 Day-1,环境泛化基线供 Day-2 大对比表复用。

| 实验 | 卡 | 进度(滚动) | 角色 |
|---|---|---|---|
| **chunk64 训练**(4 块,LaCT 大块) | GPU1 | ~14.9k/30k,loss 2.10,剩 ~5.8h | chunk 扫描点之一 → Day1.1 eval |
| **chunk256 训练**(1 块,1 更新/帧) | GPU0 | ~14.1k/30k,loss 2.17,剩 ~5.8h | chunk 扫描点之一 → Day1.1 eval |
| **TTT-16 环境泛化 eval**(LIBERO-plus 维度1-5,1627 任务) | GPU2/3 | shard0=41、shard1=43 / 各 ~810 | Day2.4 大对比表的 TTT 泛化基线 |

- chunk 扫描已有点:16 块=97.80%(完成)。+ 4 块 / 1 块(本阶段训练中)→ Day1.1 出完整 16/4/1 曲线。
- 环境泛化:TTT-16 跑完后,attention 基线的同款环境泛化 eval 进 Day-2,二者合成 TTT vs attn 逐维度泛化表。
- Cron `c3a11114`(:08/:38)自动检查 chunk 训完触发基础 spatial eval(需 ≥4 空闲卡,见卡冲突 ⚠️)。

---

## 背景:已确认的关键事实(无需再调研)

- **VLANeXt 两个 expert**:action DiT(`policies.py` MoEBlock,29 层,8 个 action token,**当前双向** `ttt_causal=False`)、
  vision/video DiT(`generator.py` MoEGeneratorBlock,29 层,256 个 image token,**因果** `causal=True`)。
- **现有 mixer 分派**:`_mixer_at(layer_idx, depth, mixer_type, mix_every_n)`,`mixer_type ∈ {attention, ttt}`,
  默认 `mix_every_n=4`(每 4 层第 4 层换 mixer)。新增 mixer 只需扩展这个枚举 + 在 MoEBlock/MoEGeneratorBlock
  里加一个构造分支,**外壳(adaLN/残差/MLP/norm)完全不动**(方式 B 语义一致)。
- **GatedDeltaNet 已 vendored**:`src/models/fla/layers/gated_deltanet.py`(+ test_gated_deltanet.py 已通过)。
  参数量 ≈ 6·d²/层,`mode='chunk'`,**固有因果**(无法做 action 的双向语义)。
- **GLA**:在 kairos `third_party/fla/layers/gla.py`,也是 fla 系线性注意力,同样**因果**。KairosDiT 实际用的是
  GatedDeltaNet,但 GLA 是同族、接口类似(hidden_size/num_heads/mode='chunk'/use_gate)。
- **KairosDiT block 结构**(kairos_dit.py:632-790):self_attn_norm → (use_linear_attn ? GatedDeltaNet : SelfAttention)
  → cross_attn(注入 context)→ ffn,全部 adaLN 调制(`modulation` 6 chunk)+ GateModule 残差。
  线性注意力层只替换 self-attention 那一路,cross-attention 仍是 softmax —— **这点很重要:context(VLM)注入靠
  cross-attn,不靠把 VLM 塞进线性层的 KV**。这与我们 TTT 方式 B(把 VLM 拼进 update KV)是两种不同注入哲学,
  集成时需明确选哪种(见 Day-2 决策点)。

---

## 资源现状(滚动更新)

- GPU0=chunk256 训练、GPU1=chunk64 训练(各剩 ~6h,约今天中午完)。
- GPU2/3=TTT-16 LIBERO-plus 环境泛化 eval(~53h,要到 ~6/9 才完)。
- ⚠️ **卡冲突**:chunk 训练完后只空出 GPU0/1(2 张),而 `auto_eval_chunks.sh` 要 ≥4 张才触发 →
  需手动决定:暂停环境泛化腾卡跑 chunk eval(4 卡 ~1h),还是 2 卡慢跑。

---

## Day 1(6/07)— chunk eval 收尾 + GLA 层落地

### 1.1 chunk64 / chunk256 训完后 eval(成功率扫描收尾)
- 等 `checkpoint_final.pt`(约中午)。决策腾卡方案(见上 ⚠️)。
- 跑 4 分片 LIBERO-spatial eval → 补全 **16 块 / 4 块 / 1 块** 成功率曲线,对照 TTT-16=97.80%、attn=97.60%。
- 产出:chunk-size→成功率曲线(验证 LaCT 大块在精度上是否站得住)。

### 1.2 实现 GLA mixer(参考 KairosDiT,方式 B 外壳不变)
- 把 GLA 层 vendor 进 `src/models/fla/layers/gla.py`(+ 依赖闭包,与 gated_deltanet 一致的做法)。
- 在 `ttt.py` 同级新增/复用 mixer 注册:`mixer_type` 枚举加 `"gla"`、`"gdn"`(GatedDeltaNet)。
- MoEBlock / MoEGeneratorBlock 各加构造分支:`mixer_type=='gla'` → 用 GLA 层替换内部 token-mixing 调用,
  **保留 adaLN/残差/MLP**。
- **context 注入策略**:已锁定 **方式 B**(见顶部"已锁定决策")——把 VLM 投影后**拼进 GLA 的输入序列**
  当前缀 token,与现有 TTT 一致。**不**加 KairosDiT 式独立 cross-attn(那样改动大且消融不可比)。
  好处:TTT vs GLA vs GatedDeltaNet 用同一种 VLM 注入,只换 token-mixing 算子,差异完全归因于算子本身。
- 单元测试:`test_gla_mixer.py` —— 形状/finite/grad;因果版扰动测试(image token p 扰动 → <p 输出不变)。

### 1.3 action DiT 因果 TTT 变体(因果性消融的第一步)
- 现在 action 是 `ttt_causal=False`(双向)。加一个 config 开关 `policy_ttt_causal`,允许 action 用**因果 TTT**
  (复用 vision 已验证的 `causal_block_fast_weight_swish_glu`,chunk_size 取 1 或 2,因为只有 8 个 action token)。
- 这是**廉价的因果性探针**:同一个 TTT 算子,仅切换 causal 标志,直接量出"动作 token 加因果约束"对成功率的影响。
- 单测:确认 8-token 因果 TTT forward/backward 正常,且扰动 action token t → <t 输出不变。

---

## Day 2(6/08)— 因果性消融训练 + GLA/GDN 对比

### 2.1 训练:action 双向 TTT vs action 因果 TTT(核心因果性消融)
- 用与 chunk16 完全相同的配置,只切 `policy_ttt_causal: false → true`,30k 步,world-model。
- **科学问题**:动作生成是否需要双向?diffusion 联合去噪 8 个动作本应双向,若强加因果掉点 →
  说明"把 action DiT 换成 GatedDeltaNet(因果)不合理";若不掉点 → 因果结构可接受,GatedDeltaNet 可上。
- 这是判断"action 换 Gated DeltaNet 是否合理"的**直接判据**,先用便宜的 TTT-causal 探针验证,再决定要不要费力上 GDN。

### 2.2 训练:GLA mixer(action+vision 都换 GLA)
- `mixer_type='gla'`,[A,A,A,GLA],30k 步,world-model。
- 对照 TTT / attention,看 GLA 这种"门控线性注意力"在 VLA 上的成功率与训练稳定性。

### 2.3(条件触发)GatedDeltaNet mixer
- 若 2.1 的因果探针表明"action 加因果不掉点",则把 action+vision 换 GatedDeltaNet(`mixer_type='gdn'`)训练,
  与 KairosDiT 路线完全对齐;否则只在 **vision**(本就因果)上用 GDN,action 保持双向 TTT/attention。

### 2.4 汇总
- 统一 eval(LIBERO-spatial 基础 + 选取维度的环境泛化),产出大对比表:
  **attention / TTT(双向) / TTT(因果) / GLA / GatedDeltaNet** × {成功率, 逐维度泛化, 训练速度, 参数量}。

---

## 关键决策点

1. **chunk eval 腾卡方案**(待用户拍板):暂停环境泛化 vs 2 卡慢跑 vs 等 2 天。
2. ~~GLA/GDN 的 VLM 注入~~ → **已锁定方式 B**(拼 KV,对齐 TTT)。
3. **GatedDeltaNet 是否全上**:取决于 2.1 因果探针结果,先做探针再定。

## 风险

- GLA/GatedDeltaNet 的 triton kernel 在隔离 venv(triton 3.2 / torch 2.6)下的兼容性 —— gated_deltanet 已过测,
  GLA 需同样验证(可能要 vendor 额外 ops 闭包)。
- action 只有 8 token,线性注意力/因果分块的收益本就小,结论可能是"动作端用线性注意力无明显优势"——
  这本身也是有价值的负结果。
- @torch.compile 在 fla kernel + 变长 AR 推理下的 recompile storm(已知,环境泛化 eval 用 EVAL_WM_NO_COMPILE=1 规避)。
