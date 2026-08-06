#!/usr/bin/env bash
# Resume attn main training from checkpoint_50000.pt on a remote 4-GPU box.
# Continues to max_steps=60000 (~9.4k steps left). 4 GPUs x per-device batch 4 = global 16,
# matching the original attn run exactly. Requires shared /mnt/afs-h200 access on the remote box.
#
# Usage:
#   export WANDB_API_KEY=<your key>     # optional; or set use_wandb:false in the config
#   bash scripts/run_attn_resume50k_4gpu.sh
#
# Override GPUs / port if needed:
#   CUDA_VISIBLE_DEVICES=4,5,6,7 MASTER_PORT=29642 bash scripts/run_attn_resume50k_4gpu.sh
set -u
cd /mnt/afs-h200/yuyangcheng/workplace/VLANeXt
REPO=$PWD

CKPT="$REPO/VLANeXt_robolab_attn/molmoact_droid/checkpoint_50000.pt"
if [ ! -f "$CKPT" ]; then
  echo "ERROR: resume checkpoint not found: $CKPT" >&2
  echo "       (is the shared /mnt/afs-h200 mounted on this box?)" >&2
  exit 1
fi

# WANDB_API_KEY is read from the environment (NOT hardcoded). If the config has
# use_wandb: true and no key is set, warn but continue (wandb will run offline/disabled).
if grep -qE '^\s*use_wandb:\s*true' config/ablation_wm_attention_molmoact_droid_resume50k.yaml \
   && [ -z "${WANDB_API_KEY:-}" ]; then
  echo "WARN: use_wandb=true but WANDB_API_KEY is unset. Export it first, or set use_wandb:false." >&2
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTHONPATH="$REPO"
export TORCHDYNAMO_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
VENV=/mnt/afs-h200/yuyangcheng/venvs/fla_triton32/bin
NGPU=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)
LOG="logs/attn_resume50k_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs

echo "Resuming attn from $CKPT on $NGPU GPU(s) [$CUDA_VISIBLE_DEVICES] -> $LOG"
nohup $VENV/torchrun --nproc_per_node="$NGPU" --master_port="${MASTER_PORT:-29641}" \
  scripts/train.py --config config/ablation_wm_attention_molmoact_droid_resume50k.yaml \
  > "$LOG" 2>&1 &
echo "launched PID $! -> $LOG"
echo "verify:  tail -f $LOG   (expect 'Resuming ... checkpoint_50000.pt' and first loss at step ~50000)"
