#!/usr/bin/env bash
# Retrain GDN (GatedDeltaNet mixer) — previous run DIVERGED at lr 1e-4 (loss 4.25->14.5).
# Fixes in config/ablation_wm_gdn.yaml: lr 1e-4->5e-5, warmup 1500->3000, max_grad_norm 1.0->0.5.
# 2-GPU DDP (standing rule). GPU0/1.
set -u
cd /mnt/afs-h200/yuyangcheng/workplace/VLANeXt
VENV=/mnt/afs-h200/yuyangcheng/venvs/fla_triton32/bin

export WANDB_API_KEY="$WANDB_API_KEY"
export TORCHDYNAMO_DISABLE=1
export CUDA_VISIBLE_DEVICES=0,1
export PYTHONPATH=/mnt/afs-h200/yuyangcheng/workplace/VLANeXt:${PYTHONPATH:-}

$VENV/torchrun --nproc_per_node=2 --master_port=29531 \
  scripts/train.py --config config/ablation_wm_gdn.yaml
