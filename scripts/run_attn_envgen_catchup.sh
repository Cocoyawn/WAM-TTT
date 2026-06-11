#!/usr/bin/env bash
# Catch-up env-gen eval for ATTENTION baseline: the original 5 shards only covered
# task_ids 0-1409 (1020 tasks); the env-gen full set is 1627 (0-2401). This launches
# 8 workers for the missing 607 tasks (1410-2401, all Noise+Light) as shard5..12.
# RAM-bound: stagger launches ~40s; 8 workers x ~22GB ~= 176GB on top of running jobs.
set -u
cd /mnt/afs-h200/yuyangcheng/workplace/VLANeXt
REPO=$PWD
PLUS_DIR=third_party/LIBERO-plus
VENV=/mnt/afs-h200/yuyangcheng/venvs/fla_triton32/bin/python

export WANDB_MODE=disabled
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa TORCHDYNAMO_DISABLE=1
export LIBERO_CONFIG_PATH=/tmp/libero_plus_cfg

# round-robin 8 workers over 4 GPUs (2 per card)
GPUS=(0 1 2 3 0 1 2 3)
i=0
for s in 5 6 7 8 9 10 11 12; do
  g=${GPUS[$i]}
  log=$REPO/logs/plus_envgen_attn_shard${s}.log
  echo "launch shard$s on GPU$g -> $log"
  CUDA_VISIBLE_DEVICES=$g PYTHONPATH="$REPO/$PLUS_DIR:$REPO" \
    nohup bash -c "cd $REPO/$PLUS_DIR && $VENV $REPO/scripts/libero_plus_bench_eval.py --config $REPO/config/libero_plus_envgen_attn_shard${s}.yaml" \
    > "$log" 2>&1 &
  i=$((i+1))
  sleep 40   # stagger to avoid simultaneous model-load RAM/GPU spikes
done
echo "all 8 catch-up workers launched"
