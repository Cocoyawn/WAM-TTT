---
name: vlanext-plot-loss
description: Plot VLANeXt training loss curves as ASCII text in the terminal — no wandb or browser needed. Use when the user asks to "plot loss", "draw the loss curve", "show loss curve", "compare loss curves", "看loss曲线", "对比曲线", or wants to see a training run's loss trajectory (especially while a run is still training, since wandb hides history for running runs). Supports overlaying multiple runs for divergence comparison.
---

# VLANeXt plot loss (ASCII)

Render training-loss curves as text — the immediate way to inspect a run's loss, including **while it's
still training** (wandb only exposes history after a run finishes; this reads the local log directly).

Repo: `/mnt/afs-h200/yuyangcheng/workplace/VLANeXt`
Venv: `/mnt/afs-h200/yuyangcheng/venvs/fla_triton32/bin/python`

## Usage

```bash
cd /mnt/afs-h200/yuyangcheng/workplace/VLANeXt
PY=/mnt/afs-h200/yuyangcheng/venvs/fla_triton32/bin/python

$PY scripts/plot_loss.py                      # list all runs (logs/*.log) + final loss
$PY scripts/plot_loss.py gdn_retrain          # single run curve
$PY scripts/plot_loss.py gdn_retrain gdn      # OVERLAY 2+ runs (divergence comparison)
$PY scripts/plot_loss.py --rows 30 ttt_chunk256   # taller plot
```

- A run `key` is matched as a **substring of the log basename** (`logs/ablation_wm_<key>.log`).
  E.g. `gdn_retrain` → the lr-5e-5 fix run; `gdn` → the diverged run-1 (note: `gdn` also matches
  `gdn_retrain`; the script picks the first/exact and prints which log it used — pass the fuller key
  to disambiguate).
- Reads `loss=<float>` from each log line, auto-dedups DDP double-logging, samples ~24 points.
- Single run → vertical bar chart (bar length ∝ loss, auto-scaled to its own range).
- **Multiple runs → outputs each run's FULL curve separately on a SHARED y-scale** (so the two bar
  charts are directly comparable point-for-point), **followed by** an overlaid `[A]/[B]` scatter
  (left=low loss, right=high; top→bottom = progress 0→100%) for shape-at-a-glance.
  Always show the user BOTH the separate per-run curves and the overlay when comparing.

## When to use vs wandb
- **Running run / immediate look** → this skill (wandb history is empty until a run finishes — known
  wandb 0.27.1 behavior; see `vlanext-wandb` skill notes).
- **Polished curves / finished run / lr & grad_norm too** → wandb (`vlanext-wandb` skill gives the URL).

## Common asks
- "GDN 现在的 loss 曲线" → `plot_loss.py gdn_retrain`
- "对比炸的和修好的" → `plot_loss.py gdn_retrain gdn` (shows the diverged run climbing to ~14.5 vs the
  fix staying low)
- "所有训练 loss 对比" → list first, then overlay the relevant keys.
