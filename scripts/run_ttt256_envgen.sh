#!/usr/bin/env bash
# TTT-256 (vision chunk_size=256) env-gen generalization eval — 8-way, full 1627 tasks.
# chunk256 deploy inference is ~2.3x faster than chunk16, so 8-way should finish well under 1 day.
# ckpt = ttt_chunk256_libero_spatial/checkpoint_final.pt (chunk_size read from ckpt config automatically).
# RAM-bound: 8 workers x ~22GB. GDN training is on GPU0/1; pack eval mostly on GPU2/3, lightly on 0/1.
set -u
cd /mnt/afs-h200/yuyangcheng/workplace/VLANeXt
REPO=$PWD
PLUS_DIR=third_party/LIBERO-plus
VENV=/mnt/afs-h200/yuyangcheng/venvs/fla_triton32/bin/python

export WANDB_MODE=disabled
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa TORCHDYNAMO_DISABLE=1
export LIBERO_CONFIG_PATH=/tmp/libero_plus_cfg

# GPU0/1 already carry GDN training (~41GB each); put 1 eval worker each there, 3 each on free GPU2/3.
GPUS=(2 3 2 3 2 3 0 1)
i=0
for s in 0 1 2 3 4 5 6 7; do
  g=${GPUS[$i]}
  log=$REPO/logs/plus_envgen_ttt256_shard${s}.log
  echo "launch ttt256 shard$s on GPU$g -> $log"
  CUDA_VISIBLE_DEVICES=$g PYTHONPATH="$REPO/$PLUS_DIR:$REPO" \
    nohup bash -c "cd $REPO/$PLUS_DIR && $VENV $REPO/scripts/libero_plus_bench_eval.py --config $REPO/config/libero_plus_envgen_ttt256_shard${s}.yaml" \
    > "$log" 2>&1 &
  i=$((i+1))
  sleep 40   # stagger model loads (RAM/GPU spike protection)
done
echo "all 8 ttt256 catch-up workers launched"
