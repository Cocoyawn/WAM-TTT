# WAM-TTT: Token-Mixer Ablations for VLA World Models

A research fork of **[VLANeXt](https://github.com/DravenALG/VLANeXt)** (ICML 2026) that ablates the
**token mixer** inside the vision world-model DiT — comparing **softmax attention**, **TTT**
(test-time-training fast-weight), **GatedDeltaNet (GDN)**, and **sliding-window attention (SWA)** — and
measures their **robustness under environment perturbations** on LIBERO / LIBERO-plus.

> This repo is the **code + experiment-tooling backup** of the work. The ~400 GB of checkpoints,
> rendered videos, and benchmark assets are intentionally **not** tracked (see [What's *not* here](#whats-not-in-this-repo)).
> Everything you need to *re-run* the experiments is here.
>
> Base codebase, models, and benchmarks are **VLANeXt** by Xiao-Ming Wu et al. (S-Lab NTU; SYSU; ACE
> Robotics) — see [Citation & credit](#citation--credit). This fork adds the mixer ablation, the job
> scheduler, the env-gen severity analysis, and the tooling described below.

---

## TL;DR — what this fork adds

| Area | What | Where |
|---|---|---|
| **Mixer ablation** | TTT / GDN / SWA / SWA+GDN / SWA+TTT mixers in the generator DiT | `src/models/{ttt,linear_attn_mixer,generator}.py` |
| **Block-causal TTT** | `chunk_size` controls TTT block granularity (16 blocks → 1 block/frame) | `generator_ttt_chunk_size` in configs |
| **Job scheduler** | concurrent, fault-tolerant GPU/RAM orchestrator for *all* trainings + evals | `scripts/scheduler.py` |
| **Env-gen eval** | sharded LIBERO-plus generalization eval (1627 tasks, 5 perturbation dims) | `scripts/libero_plus_bench_eval.py`, `config/libero_plus_envgen_*` |
| **Severity analysis** | re-aggregate results by `difficulty_level` 1-5 → robustness-vs-strength curves | `scripts/plot_severity_curves.py`, `scripts/aggregate_envgen.py` |

**Headline result:** TTT's fast-weight online adaptation gives a **robustness-slope** advantage that
*widens with perturbation strength*. At the hardest sensor-noise level TTT holds ~100% success while
softmax attention collapses to ~33%. See [Key results](#key-results).

---

## Repository layout

```
src/
  models/
    VLANeXt.py            # top-level VLA: encoder → connector → generator(world model) → policy
    generator.py          # the vision DiT; mixer dispatch + SWA / block-causal masks
    ttt.py                # TTT fast-weight mixer (block-causal, chunk_size)
    linear_attn_mixer.py  # GatedDeltaNet / linear-attention mixers
    test_*.py             # unit + e2e tests — good entry points to understand each module
  datasets/               # libero_act / droid_act dataloaders
  evaluation/             # libero_bench + libero_plus_bench eval harnesses
config/                   # 90 YAML configs — one per training run / eval shard (see below)
scripts/
  train.py                # single entry point for ALL trainings (torchrun DDP)
  scheduler.py            # resource-managed job runner (preferred launch path)
  libero_plus_bench_eval.py   # env-gen eval worker (one shard)
  aggregate_envgen.py     # results → per-dimension table (handles 3 known pitfalls)
  plot_severity_curves.py # results → robustness-vs-difficulty curves
  plot_loss.py            # ASCII loss curve from a live log (no wandb needed)
  exp_status.py           # disk/RAM/GPU + all run statuses at a glance
docs/                     # result figures (severity_curves.png, attn_mask_comparison.png)
claude_assets/            # project memory notes + experiment-management skills (see below)
*.md                      # design docs: TTT_ARCHITECTURE, DESIGN_SPACE, RESULTS, COMMON_ISSUES
```

---

## Environment setup

```bash
conda create -n VLANeXt python=3.10 && conda activate VLANeXt
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
conda install -c conda-forge ffmpeg
```

**Benchmarks** (not vendored — clone into `third_party/`):
```bash
cd third_party
# LIBERO
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git && cd LIBERO && pip install . && cd ..
# LIBERO-plus (robustness benchmark; we pinned HEAD 4976dc3)
git clone https://github.com/sylvestf/LIBERO-plus.git && cd LIBERO-plus && pip install .
apt install libexpat1 libfontconfig1-dev libpython3-stdlib libmagickwand-dev
pip install -r extra_requirements.txt
conda env config vars set LIBERO_CONFIG_PATH=~/.libero_plus
# then download the LIBERO-plus assets per its README
```

**Data:** OpenVLA-modified LIBERO episodes:
```bash
hf download openvla/modified_libero_rlds --repo-type dataset --local-dir LIBERO_modified
```
Set `data.data_root` in each config to your local path (ours: `~/data/LIBERO_modified/`), and
`model.pretrained_checkpoint` to your encoder/init checkpoint.

> **PYTHONPATH gotcha** (the #1 setup error): training needs `PYTHONPATH=$REPO`; LIBERO-plus eval needs
> `PYTHONPATH=$REPO/third_party/LIBERO-plus:$REPO` **and** you must `cd` into `third_party/LIBERO-plus`
> (and set `LIBERO_CONFIG_PATH`, `MUJOCO_GL=osmesa`), else `ModuleNotFoundError: libero`. The provided
> launch scripts set all of this for you.

---

## Quick start

### 1. Train a world model (single run)
```bash
PYTHONPATH=$PWD torchrun --nproc_per_node=4 --master_port=29505 \
    scripts/train.py --config config/ablation_wm_ttt_chunk256.yaml
```
`batch_size` in the config is the **total** across GPUs (auto-divided by `world_size`). Each run writes
`VLANeXt_ablation_wm/<save_dir>/checkpoint_final.pt`. **Keep `save_interval: 10000`** — the default 2000
fills disk (each WM checkpoint is ~20 GB).

### 2. Run the env-gen generalization eval (8-way sharded)
```bash
REPO=$PWD; cd third_party/LIBERO-plus
for s in $(seq 0 7); do
  MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa LIBERO_CONFIG_PATH=~/.libero_plus \
  PYTHONPATH=$REPO/third_party/LIBERO-plus:$REPO \
  python $REPO/scripts/libero_plus_bench_eval.py \
      --config $REPO/config/libero_plus_envgen_ttt256_shard${s}.yaml &
done; wait
```

### 3. Aggregate + plot
```bash
python scripts/aggregate_envgen.py          # per-dimension TTT-16 / TTT-256 / Attention table
python scripts/plot_severity_curves.py      # robustness vs difficulty_level → docs/severity_curves.png
python scripts/plot_loss.py swa_ttt         # live ASCII loss curve for any run
python scripts/exp_status.py                # disk/RAM/GPU + every run's status
```

### Recommended: launch everything through the scheduler
Rather than launching by hand, append a job to the `QUEUE` in `scripts/scheduler.py` and let it pack
jobs onto the GPUs with a dual **GPU-free + cgroup-RAM** gate (prevents the OOM / port-collision /
silent-skip failures that ad-hoc launches cause):

```bash
tmux new-session -d -s vlanext-sched -c $PWD
tmux send-keys -t vlanext-sched "python scripts/scheduler.py 2>&1 | tee logs/scheduler.log" Enter
python scripts/scheduler.py --status     # queue + per-job status
```
Two job kinds: `train` (N-GPU DDP, `gpus` configurable) and `eval` (N single-GPU shard workers that
co-reside on cards and elastically scale to free RAM). Details in `claude_assets/skills/vlanext-scheduler`.

---

## Understanding the configs

Configs are named `ablation_wm_<mixer>[_<chunk>][_<suite>].yaml`. The mixer is chosen by a few fields:

```yaml
model:
  generator_mixer_type: "ttt"        # attention | ttt | gdn | swa
  generator_mix_every_n: 4           # every 4th DiT layer is a mixer layer (→ 7 of 29 layers)
  generator_ttt_chunk_size: 256      # TTT block size: 16 → 16 blocks/frame; 256 → 1 block/frame
  generator_fallback_mixer: "swa"    # (SWA+X variants) non-mixer layers use sliding-window attn
  generator_swa_window_size: 64      # SWA causal window in tokens
train:
  distributed: true                  # REQUIRED for multi-GPU — false makes all ranks land on GPU0
  batch_size: 16                     # TOTAL across GPUs
  warmup_steps: 1500
data:
  max_steps: 30000                   # 30k for spatial/object/goal; ×1.2 for long
                                     # NOTE train.py reads data.max_steps (not train.max_steps)
```

> **GDN stability:** GatedDeltaNet diverges at lr 1e-4 (loss → 14.5). Use **lr 5e-5, warmup 3000,
> grad_clip 0.5** (the `*_gdn_retrain` recipe) — converges to ~5.6.

A guided tour of the full **12 design spaces** from the base paper is in
[DESIGN_SPACE.md](DESIGN_SPACE.md); mixer internals in [TTT_ARCHITECTURE.md](TTT_ARCHITECTURE.md);
known pitfalls in [COMMON_ISSUES.md](COMMON_ISSUES.md).

---

## Key results

LIBERO-spatial trained, evaluated on **LIBERO-plus env-gen (1627 tasks, 5 perturbation dimensions)**.

**Per-dimension success rate (all configs aligned):**

| Dimension | TTT-16 | **TTT-256** | Attention |
|---|---|---|---|
| Camera | 69.1% | **78.7%** | 70.2% |
| Light | 99.0% | 99.0% | 98.6% |
| Background | 96.5% | 97.7% | 98.1% |
| Noise | 98.0% | 95.4% | 80.9% |
| Robot | 80.3% | **82.9%** | 69.4% |
| **Total** | 87.5% | **89.9%** | 81.9% |

**Robustness vs. perturbation strength** (`difficulty_level` 1→5, success-rate drop; smaller = more robust):

| Dim | TTT-16 | TTT-256 | Attention |
|---|---|---|---|
| Noise | 100→100 (**0**) | 96→75 (−21) | 90→**33** (**−57**) |
| Robot | 97→60 (−37) | 99→53 (−46) | 97→**29** (**−68**) |
| Camera | 81→82 (+1) | 96→95 (0) | 88→68 (−20) |

→ The TTT advantage is **concentrated at high severity**: at clean/low levels everyone ties; TTT's
fast-weight online adaptation degrades far more gracefully as the perturbation strengthens. This is the
core evidence for the online-adaptation claim. Figure: `docs/severity_curves.png`.

---

## What's *not* in this repo

Excluded via `.gitignore` to keep the backup small (~85 MB vs ~400 GB on disk):

- **Checkpoints / model outputs** — `VLANeXt_ablation_wm/`, `VLANeXt_final_libero/`, all `*.pt`
- **Benchmarks** — `third_party/` (clone from upstream; LIBERO-plus pin HEAD `4976dc3`)
- **Demo videos** — `docs/static/`, all `*.mp4`
- **W&B run dirs** — `wandb/`

Training/eval **text logs** (`logs/`) *are* included so loss/eval trajectories are reproducible.

---

## `claude_assets/` — experiment-management tooling

This project was run with an AI-assisted experiment loop. `claude_assets/` versions the durable parts:
- **`memory/`** — project notes: resource limits, GDN stability recipe, env-gen results, conventions.
  A condensed lab notebook.
- **`skills/`** — reusable procedures: `vlanext-scheduler` (job orchestration), `vlanext-experiments`
  (status/launch), `vlanext-plot-loss`, `vlanext-wandb`, `vlanext-backup`.

Documentation/automation only — the experiments run fine without them.

---

## Citation & credit

This is a research fork. The base codebase, models, and benchmark are **VLANeXt** (ICML 2026) by
Xiao-Ming Wu, Bin Fan, Kang Liao, Jian-Jian Jiang, Runze Yang, Yihang Luo, Zhonghua Wu, Wei-Shi Zheng,
and Chen Change Loy (S-Lab NTU; Sun Yat-sen University; ACE Robotics):

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
[LIBERO-plus](https://github.com/sylvestf/LIBERO-plus). Please cite and follow their licenses
(base project: NTU S-Lab License 1.0, see [LICENSE](LICENSE)).
