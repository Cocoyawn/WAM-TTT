# 隔离环境说明 (TTT / GatedDeltaNet 消融实验)

为在**不污染系统环境**的前提下运行 VLANeXt + 自研 TTT 层 + GatedDeltaNet 基线,
我们建立了一个独立 venv。系统环境(定制 torch 2.3 + triton 3.0 + transformers 4.41)
**完全未改动**。

## venv 位置
```
/mnt/afs-h200/yuyangcheng/venvs/fla_triton32   (约 5.7G)
```

## venv 与系统的差异

| 包 | 系统 (dist-packages) | venv (隔离, shadow 系统) |
|---|---|---|
| torch | 2.3.0a0+nv24.04 | **2.6.0+cu124** |
| triton | 3.0.0 | **3.2.0** |
| transformers | 4.41.2 (无 Qwen3-VL) | **5.1.0** |
| diffusers | 缺/旧 | **0.36.0** |
| accelerate / peft / tokenizers | 旧 | 1.10.1 / 0.19.1 / 0.22.2 |
| scipy/sklearn/pandas/pyarrow/soxr | numpy1.x 编译(ABI 冲突) | 重装为 numpy2 兼容版 |

> venv 用 `uv venv --system-site-packages` 创建,继承系统的 numpy/CUDA 驱动等,
> 仅 shadow 上述需要新版的包。GatedDeltaNet 的 fla triton kernel 需要 triton>=3.2,
> 而 transformers 5.1 / diffusers 0.36 需要 torch>=2.4,故 venv 内升级了 torch+triton。

## 重建步骤(若 venv 损坏)
```bash
VENV=/mnt/afs-h200/yuyangcheng/venvs/fla_triton32
uv venv --system-site-packages --python /usr/bin/python3 $VENV
# torch 2.6 + cu124 (匹配系统 CUDA 12.4 驱动 550.x),自带 triton 3.2
VIRTUAL_ENV=$VENV uv pip install --python $VENV/bin/python \
  --index-url https://download.pytorch.org/whl/cu124 \
  torch==2.6.0+cu124 torchvision==0.21.0+cu124
# transformers 栈 (--no-deps 避免再拖动 torch)
VIRTUAL_ENV=$VENV uv pip install --python $VENV/bin/python --no-deps \
  transformers==5.1.0 diffusers==0.36.0 tokenizers==0.22.2 safetensors==0.7.0 \
  accelerate==1.10.1 "peft>=0.17.0" regex soxr
# 修 numpy1.x ABI 冲突的系统包 (shadow 成 numpy2 兼容版)
VIRTUAL_ENV=$VENV uv pip install --python $VENV/bin/python --no-deps \
  "scipy>=1.13" "scikit-learn>=1.5" joblib threadpoolctl \
  "pandas>=2.2" python-dateutil pytz tzdata "pyarrow>=17"
```

## 运行测试
```bash
cd /mnt/afs-h200/yuyangcheng/workplace/VLANeXt
VENV=/mnt/afs-h200/yuyangcheng/venvs/fla_triton32

# 1) TTT 层单元测试 (因果/双向语义 + 反向)
TORCHDYNAMO_DISABLE=1 $VENV/bin/python src/models/test_ttt.py

# 2) GatedDeltaNet 冒烟测试 (训练+推理路径)
bash src/models/run_fla_smoke.sh

# 3) 整模型端到端 (Qwen3-VL backbone + action expert, 构造+train fwd/bwd+predict)
#    跑两遍: policy_mixer_type='attention'(基线) 和 'ttt'(消融)
TORCHDYNAMO_DISABLE=1 PYTHONPATH="$PWD" $VENV/bin/python -m src.models.test_e2e_vlanext
```

## TTT 消融开关 (config: model 段)

```yaml
# action expert (policies.py 的 MoE 系列): "attention"(基线) | "ttt"(双向 TTT)
policy_mixer_type: "attention"
policy_mix_every_n: 4          # [A,A,A,T]: 每 4 层第 4 层为 TTT
# vision expert (generator.py): "attention" | "ttt"(因果 TTT)
generator_mixer_type: "attention"
generator_mix_every_n: 4
generator_ttt_chunk_size: 16
```
默认全 "attention" = 现行基线,零行为变化。改 "ttt" 即启用消融。
方式 B 语义一致:TTT 层的 update 阶段把 VLM hidden state 作为额外 k/v(action 双向
拼接;vision 因果情形下作全局非因果预更新,保住 image→VLM 全可见 + image→image 因果)。

## 已验证 (全部通过)
- TTT 层: 7/7 (因果性硬验证 / 双向性 / 反向 / 状态链式) + 方式B注入(ctx 双向/因果全局)
- GatedDeltaNet: import / chunk 训练 fwd+bwd / fused_recurrent 推理 / use_gate=False
- block 级: action(diffusion MoE)/ vision(generator) × {attention, ttt} 全 fwd+bwd
- 整模型: attention 基线 + ttt 消融 两条路径均 构造/train fwd-bwd/predict_action 通过

## 注意
- `TORCHDYNAMO_DISABLE=1` 仍建议带上(规避 inductor 边角问题;fla 的 @torch.compile
  小函数退化为 eager,数值等价)。
- Emu3.5 VisionTokenizer 仅有 config 无权重,故 `future_image_loss_weight=0` 时不加载;
  若要验证 vision expert 的图像生成损失,需补全该权重。
- 真实多卡训练(deepspeed 等)未在此 venv 验证,仅验证单卡 构造/前向/反向/推理。
