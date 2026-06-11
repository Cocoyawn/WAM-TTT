# WAM-TTT: Token-Mixer Ablations for VLA World Models

A research fork of **[VLANeXt](https://github.com/DravenALG/VLANeXt)** (ICML 2026) that ablates the
**token mixer** in the vision world-model DiT — softmax attention, **TTT** (test-time-training
fast-weight), **GatedDeltaNet (GDN)**, and **sliding-window attention (SWA)** — and quantifies their
**robustness under environment perturbations** on LIBERO / LIBERO-plus.

This repository is the **code and tooling backup** of the work; checkpoints, rendered videos, and
benchmark assets (~400 GB) are excluded (see [Exclusions](#exclusions)). All material required to
reproduce the experiments is present.

## Contributions

| Component | Description | Location |
|---|---|---|
| Mixer ablation | TTT / GDN / SWA / SWA+GDN / SWA+TTT in the generator DiT | `src/models/{ttt,linear_attn_mixer,generator}.py` |
| Block-causal TTT | `chunk_size` sets TTT block granularity (16 blocks → 1 block/frame) | `generator_ttt_chunk_size` |
| Job scheduler | Fault-tolerant GPU/RAM orchestrator for all trainings and evals | `scripts/scheduler.py` |
| Env-gen eval | Sharded LIBERO-plus generalization eval (1627 tasks, 5 dimensions) | `scripts/libero_plus_bench_eval.py` |
| Severity analysis | Re-aggregation by `difficulty_level` 1–5 → robustness curves | `scripts/{aggregate_envgen,plot_severity_curves}.py` |

**Principal finding.** TTT's online adaptation yields a robustness advantage that *widens with
perturbation strength*: at the highest sensor-noise level TTT retains ≈100% success while softmax
attention falls to ≈33% (see [Results](#results)).

## Setup

```bash
conda create -n VLANeXt python=3.10 && conda activate VLANeXt
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt && pip install flash-attn --no-build-isolation
```

Benchmarks are not vendored — clone into `third_party/`:

```bash
cd third_party
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git      && pip install -e LIBERO
git clone https://github.com/sylvestf/LIBERO-plus.git                && pip install -e LIBERO-plus  # pin 4976dc3
```

Download the OpenVLA-modified LIBERO episodes and the LIBERO-plus assets per their READMEs, then set
`data.data_root` and `model.pretrained_checkpoint` in each config.

> **PYTHONPATH.** Training requires `PYTHONPATH=$REPO`. LIBERO-plus eval requires
> `PYTHONPATH=$REPO/third_party/LIBERO-plus:$REPO`, a `cd` into `third_party/LIBERO-plus`, and
> `MUJOCO_GL=osmesa` — otherwise `ModuleNotFoundError: libero`. The launch scripts set these.

## Usage

```bash
# Train (batch_size is the total across GPUs; keep save_interval: 10000)
PYTHONPATH=$PWD torchrun --nproc_per_node=4 scripts/train.py --config config/ablation_wm_ttt_chunk256.yaml

# Env-gen eval, 8-way sharded (run from third_party/LIBERO-plus)
python scripts/libero_plus_bench_eval.py --config config/libero_plus_envgen_ttt256_shard${s}.yaml

# Aggregate and plot
python scripts/aggregate_envgen.py            # per-dimension table
python scripts/plot_severity_curves.py        # robustness vs. difficulty → docs/severity_curves.png
```

For multi-job runs, append to the `QUEUE` in `scripts/scheduler.py` and launch the scheduler under
`tmux`; it packs jobs across GPUs under a combined GPU-free and cgroup-RAM gate. Status via
`python scripts/scheduler.py --status`.

## Configuration

Configs are named `ablation_wm_<mixer>[_<chunk>][_<suite>].yaml`. The mixer is selected by:

```yaml
model:
  generator_mixer_type: "ttt"        # attention | ttt | gdn | swa
  generator_mix_every_n: 4           # every 4th DiT layer is a mixer layer (7 of 29)
  generator_ttt_chunk_size: 256      # 16 → 16 blocks/frame; 256 → 1 block/frame
  generator_fallback_mixer: "swa"    # SWA+X variants
  generator_swa_window_size: 64
train:
  distributed: true                  # required for multi-GPU; false collapses all ranks onto GPU0
data:
  max_steps: 30000                   # ×1.2 for the long suite; read from data.max_steps
```

> **GDN stability.** GatedDeltaNet diverges at lr 1e-4. Use lr 5e-5, warmup 3000, grad_clip 0.5.

See [DESIGN_SPACE.md](DESIGN_SPACE.md), [TTT_ARCHITECTURE.md](TTT_ARCHITECTURE.md), and
[COMMON_ISSUES.md](COMMON_ISSUES.md) for details.

## Results

LIBERO-spatial trained; evaluated on LIBERO-plus env-gen (1627 tasks, 5 dimensions, all configs aligned).

| Dimension | TTT-16 | TTT-256 | Attention |
|---|---|---|---|
| Camera | 69.1 | **78.7** | 70.2 |
| Light | 99.0 | 99.0 | 98.6 |
| Background | 96.5 | 97.7 | 98.1 |
| Noise | 98.0 | 95.4 | 80.9 |
| Robot | 80.3 | **82.9** | 69.4 |
| **Total** | 87.5 | **89.9** | 81.9 |

Success-rate drop from difficulty level 1 to 5 (smaller is more robust):

| Dimension | TTT-16 | TTT-256 | Attention |
|---|---|---|---|
| Noise | **0** | −21 | **−57** |
| Robot | −37 | −46 | **−68** |
| Camera | +1 | 0 | −20 |

The advantage concentrates at high severity: configurations tie at low perturbation, and TTT degrades
far more gracefully as strength increases (`docs/severity_curves.png`).

## Exclusions

Excluded via `.gitignore` (~85 MB tracked vs. ~400 GB on disk): checkpoints and model outputs
(`VLANeXt_ablation_wm/`, `VLANeXt_final_libero/`, `*.pt`); benchmarks (`third_party/`); demo videos
(`docs/static/`, `*.mp4`); W&B run directories. Training and evaluation text logs (`logs/`) are retained.

`claude_assets/` versions the project notes (`memory/`) and the experiment-management procedures
(`skills/`); both are documentation and automation only.

## Citation

The base codebase, models, and benchmark are VLANeXt (ICML 2026):

```bibtex
@article{wu2026vlanext,
    title={VLANeXt: Recipes for Building Strong VLA Models},
    author={Xiao-Ming Wu and Bin Fan and Kang Liao and Jian-Jian Jiang and Runze Yang and Yihang Luo and Zhonghua Wu and Wei-Shi Zheng and Chen Change Loy},
    journal={arXiv preprint arXiv:2602.18532},
    year={2026},
}
```

Upstream: [VLANeXt](https://github.com/DravenALG/VLANeXt) ·
[LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) ·
[LIBERO-plus](https://github.com/sylvestf/LIBERO-plus). Please observe their licenses (base project:
NTU S-Lab License 1.0, see [LICENSE](LICENSE)).
