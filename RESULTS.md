# VLANeXt Mixer 消融实验结果汇总

> 任务:在 action expert + vision expert 上,用 [A,A,A,X] 交错(每 4 层第 4 层换 mixer)
> 把 cross-attention 替换为不同 token-mixing 算子,方式 B 注入(VLM hidden state 拼进 KV,
> 语义与原 attention 一致)。世界模型模式(future_image_loss=1.0)。LIBERO-spatial。
> 最后更新:2026-06-08。未测数据留空。

## 主表:各配置全维度对比

| 配置 | 块数 | 成功率(基础) | 部署延迟 | vs attn 延迟 | 训练 step(compile ON) | vs attn 训练 | 峰值显存 |
|---|---|---|---|---|---|---|---|
| **attention** 基线 | — | 97.60% | 4.87 s | 1.00×(最快) | 799 ms | 1.00× | 33.3 GB |
| **TTT chunk16** | 16 | 97.80% | 19.76 s | 0.25×(慢 4.1×) | 1030 ms | 慢 1.29× | 33.4 GB |
| **TTT chunk64** | 4 | 97.80% | 10.82 s | 0.45×(慢 2.2×) | (略快于 chunk16) | — | ≈33.4 GB |
| **TTT chunk256** | 1 | (eval 中) | 8.72 s | 0.56×(慢 1.8×) | (略快于 chunk16) | — | ≈33.4 GB |

- chunk_size = 每块 token 数;块数 = 256 / chunk_size。chunk 越大 → 块越少 → 推理越快。
- 延迟 = `predict_action` 端到端墙钟(世界模型 256 步 vision 自回归 + 5 步 action 去噪),B=1。
- 训练 step = fwd+backward+opt.step,受控 benchmark(batch 8,grad-ckpt ON,sdpa)。

## 成功率逐任务(已完成)

| 任务 | attention | TTT-16 | TTT-64 | TTT-256 |
|---|---|---|---|---|
| task0 | 100% | 100% | 98%  | 98% |
| task1 | 100% | 100% | 100% | (eval 中) |
| task2 | 98%  | 100% | 100% | |
| task3 | 98%  | 100% | 100% | |
| task4 | 96%  | 94%  | 100% | |
| task5 | 94%  | 96%  | 94%  | 94% |
| task6 | 100% | 98%  | 94%  | |
| task7 | 100% | 98%  | 98%  | |
| task8 | 94%  | 96%  | 94%  | |
| task9 | 98%  | 96%  | 100% | |
| **总计** | **97.60%** | **97.80%** | **97.80%** | (eval 中,~10h) |

## 环境泛化(LIBERO-plus,维度 1-5,仅 TTT-16 — 78% 快照,2026-06-08)

| 维度 | 扰动类型 | TTT-16(进行中) | 已评 | attention |
|---|---|---|---|---|
| Light 光照 | 光照变化 | 99.3% | 134/135(完成) | (未跑) |
| Noise 传感器噪声 | 观测噪声 | 98.8% | 252/255(近完成) | (未跑) |
| Background 背景纹理 | 桌面/背景纹理 | 96.5% | 249/258(近完成) | (未跑) |
| Robot 机器人初始位姿 | 初始关节/位姿 | 80.3% | 281/350(进行中) | (未跑) |
| Camera 相机视角 | 视角/外参 | 66.4% | 178/268(进行中) | (未跑) |
| **总计(快照)** | — | **86.4%** | 1266/1627 | (未跑) |

环境泛化 = 任务不变、只变视觉/物理环境。1627 个 task-variant,当前 12-way 分片并行,~3-4h 跑完。
**初步结论**:外观类扰动几乎无影响(光照/噪声/背景 96.5~99.3%,接近分布内 97.8%);几何类扰动
明显掉点 —— 相机视角(66.4%,掉 31pt)和初始位姿(80.3%)是两大短板,与 LIBERO-plus 论文
"VLA 对相机外参/初始状态最敏感"一致。Camera/Robot 两维尚未评完,最终数字可能微调但维度排序已定型。
attention 基线环境泛化待 TTT-16 跑完释放卡后再测。

## Image / 世界模型生成质量(TTT-16 vs attention,30k final,分布内 spatial)

共享缓存 eval 集(两模型同一批输入)。预测单帧未来帧(horizon t+8)。
loss_img = teacher-forced 图像 token 交叉熵;tok-acc = token top-1;
PSNR/SSIM/LPIPS = free-running greedy AR(256 tok)→ Emu3.5 VQ decode → 像素 vs GT。

| 指标 | TTT-16 | attention | 优胜 |
|---|---|---|---|
| loss_img ↓ | **2.05** | 3.10 | TTT |
| token top-1 acc ↑ | **0.675** | 0.449 | TTT |
| PSNR ↑ | **21.56** | 20.19 | TTT |
| SSIM ↑ | **0.790** | 0.750 | TTT |
| LPIPS ↓ | **0.130** | 0.185 | TTT |

**结论**:分布内 TTT 生成质量**全面优于 attention**(5 项指标全胜),说明 TTT 的低训练 loss
来自 vision 分支且真转化为更好的未来帧。仅 TTT-16 与 attention 测过;chunk64/256 未测。
复现:`scripts/run_wm_quality_eval.sh`。

## 已完成代码 / 待跑实验

| 实验 | 代码 | config | 状态 |
|---|---|---|---|
| GatedDeltaNet (gdn) mixer | 测试通过 | `config/ablation_wm_gdn.yaml` | 待跑(等卡) |
| action 因果 TTT 探针 | 测试通过 | `config/ablation_wm_ttt_actcausal.yaml` | 待跑(等卡) |
| GLA mixer | 未 vendor | — | 待定 |

## 关键结论(已确立)

1. **成功率**:attention / TTT-16 / TTT-64 三者打平(97.6~97.8%),**LaCT 大块不掉点**。
2. **延迟**:attention 最快;TTT 增大 chunk 显著追近(慢 4.1× → 1.8×)但**追不平** —— 残差来自
   Newton-Schulz + apply 的固有成本,chunk 减不掉。
3. **训练**:TTT 慢 1.29×,显存与 attention 持平。
4. **环境泛化(初步)**:背景扰动几乎无影响(96.5%),机器人初始位姿明显敏感(~81%),
   与 LIBERO-plus 论文"VLA 对初始状态敏感"一致。

## ⚠️ 重要 caveat(影响数据解读)

1. **延迟对比不完全公平**:部署 benchmark 关了 compile(`TORCHDYNAMO_DISABLE=1`),TTT 的
   `@torch.compile` 失效;且 attention 走高度优化的 SDPA C++/CUDA kernel,TTT 是**纯 PyTorch +
   torch.compile 原型**(无手写 triton/cuda kernel,可选融合 kernel 未 ship)。→ **TTT 延迟被低估**。
   公平对比需:① 部署也开 compile;② 理想情况给 TTT 上 triton 融合 kernel。
2. **attention 真实训练墙钟未实测**(只有受控 benchmark);此前 attention 训练用多卡/别的配置,
   wall-clock 不可直接比。
3. **attention 环境泛化未跑**;TTT-16 环境泛化仅完成约 1/3。

## 留空待测清单

- chunk256 成功率(base eval 进行中,~10h)
- TTT-16 环境泛化剩余维度:Camera / Light / Noise(~1.5 天)
- attention 环境泛化(全维度)
- compile-ON 的公平延迟重测(+ 解决变长 AR 的 recompile 问题)
- GDN / action-因果-TTT 训练 + eval
