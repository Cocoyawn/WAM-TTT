#!/usr/bin/env bash
# Resume the 5 stalled TTT-64 env-gen shards (0/3/4/5/7) that froze on the Light subset
# at 05:11 UTC (scheduler falsely marked DONE at 1475/1627). Each resume config sets
# resume_episodes/resume_successes so the eval skips already-completed tasks and continues.
# Output is APPENDED (>>) to the ORIGINAL shard log so aggregate_envgen.collect()'s
# i-th-success-line -> task_ids[i] mapping stays sequential.
set -u
cd /mnt/afs-h200/yuyangcheng/workplace/VLANeXt
REPO=$PWD
PLUS_DIR=third_party/LIBERO-plus
VENV=/mnt/afs-h200/yuyangcheng/venvs/fla_triton32/bin/python

export WANDB_MODE=disabled
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa TORCHDYNAMO_DISABLE=1
export LIBERO_CONFIG_PATH=/tmp/libero_plus_cfg

# hp_object on GPU0/1, hp_goal on GPU2/3 (~41GB free each). Eval worker ~1GB GPU + 22GB RAM.
SHARDS=(0 3 4 5 7)
GPUS=(0 1 2 3 0)
i=0
for s in "${SHARDS[@]}"; do
  g=${GPUS[$i]}
  log=$REPO/logs/plus_envgen_ttt64_shard${s}.log
  echo "resume ttt64 shard$s on GPU$g (append) -> $log"
  CUDA_VISIBLE_DEVICES=$g PYTHONPATH="$REPO/$PLUS_DIR:$REPO" \
    nohup bash -c "cd $REPO/$PLUS_DIR && $VENV $REPO/scripts/libero_plus_bench_eval.py --config $REPO/config/libero_plus_envgen_ttt64_shard${s}_resume.yaml" \
    >> "$log" 2>&1 &
  echo "  pid $!"
  i=$((i+1))
  sleep 40   # stagger model loads
done
echo "all 5 ttt64 resume workers launched"
