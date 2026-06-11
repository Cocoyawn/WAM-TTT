---
name: vlanext-wandb
description: Show wandb training-curve links for VLANeXt experiments. Use when the user asks "where is the loss curve", "wandb link", "training curve", "show me the loss", or wants to view/open a wandb run for any VLANeXt training (GDN, TTT, chunk, etc). Prints every run's wandb URL + live loss + status, dynamically discovered from logs.
---

# VLANeXt wandb runs

Give the user wandb links to view training-loss curves for VLANeXt experiments.

Repo: `/mnt/afs-h200/yuyangcheng/workplace/VLANeXt`
wandb project: `cocoyawn2035-tsinghua-university/VLANeXt_ablation_wm`

## Usage

Run the index (read-only, auto-discovers every run from `logs/*.log`):
```bash
cd /mnt/afs-h200/yuyangcheng/workplace/VLANeXt
/mnt/afs-h200/yuyangcheng/venvs/fla_triton32/bin/python scripts/wandb_runs.py
```

It prints, per training run: friendly label, **[LIVE]/done** status (by log mtime), latest loss, a note,
and the **wandb URL**. Relay the URLs to the user, highlighting the one they asked about.

## Notes
- New training runs appear automatically (no hardcoding) — the script greps each log's first wandb URL.
- **Disambiguate same-config reruns**: GDN run-1 (`nandmdn6`) DIVERGED to loss 14.5; the live retrain is
  `avz3bt50` (lr 5e-5 fix). Both share `--config ablation_wm_gdn.yaml`, so point the user at the right run id.
- Local wandb data lives under `wandb/run-<timestamp>-<id>/` if offline inspection is needed.
- For a numeric loss trajectory without opening a browser:
  `grep -oE 'loss=[0-9.]+' logs/<run>.log | awk 'NR%1000==1'`
- Related: status board is the `vlanext-experiments` skill (`scripts/exp_status.py`).
