#!/usr/bin/env bash
# Environment-generalization eval on LIBERO-plus (dims 1-5: Camera/Light/Background/
# Noise/Robot-init, 1627 spatial tasks). Shards the task-index list across the GPUs
# passed in $GPUS (default "2 3"), one shard per GPU, OSMesa render, isolated venv.
# Usage: GPUS="2 3" CKPT=<path> TAG=<name> bash scripts/eval_plus_envgen.sh
set -u
cd /mnt/afs-h200/yuyangcheng/workplace/VLANeXt
PLUS_DIR=third_party/LIBERO-plus
VENV=/mnt/afs-h200/yuyangcheng/venvs/fla_triton32/bin/python

export WANDB_API_KEY=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
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export TORCHDYNAMO_DISABLE=1
export LIBERO_CONFIG_PATH=/tmp/libero_plus_cfg

GPUS="${GPUS:-2 3}"
CKPT="${CKPT:-/mnt/afs-h200/yuyangcheng/workplace/VLANeXt/VLANeXt_ablation_wm/ttt_libero_spatial/checkpoint_30000.pt}"
TAG="${TAG:-ttt16}"
IDX_JSON=/tmp/env_dim_task_idx.json

read -ra GPU_ARR <<< "$GPUS"
NSHARD=${#GPU_ARR[@]}
echo "[envgen] ckpt=$CKPT tag=$TAG shards=$NSHARD gpus=${GPU_ARR[*]}"

# split the 1627 env-dim task indices into NSHARD round-robin shards -> JSON files
$VENV - "$IDX_JSON" "$NSHARD" "$TAG" <<'PY'
import json, sys
idx=json.load(open(sys.argv[1])); n=int(sys.argv[2]); tag=sys.argv[3]
for s in range(n):
    shard=idx[s::n]  # round-robin = balanced difficulty mix per shard
    json.dump(shard, open(f"/tmp/envgen_{tag}_shard{s}.json","w"))
    print(f"shard{s}: {len(shard)} tasks")
PY

for s in $(seq 0 $((NSHARD-1))); do
  gpu=${GPU_ARR[$s]}
  ids=$($VENV -c "import json;print(','.join(map(str,json.load(open('/tmp/envgen_${TAG}_shard${s}.json')))))")
  cfg="config/libero_plus_envgen_${TAG}_shard${s}.yaml"
  cat > "$cfg" <<YAML
data:
  augmentation:
    center_crop: true
    center_crop_ratio: 0.9
model:
  diffusion_steps: 5
  scheduler_type: "flow_match"
  attn_implementation: "sdpa"
eval:
  finetuned_checkpoint: "$CKPT"
  task_suite_name: "libero_spatial"
  task_ids: [$ids]
  shard_tag: "envgen_${TAG}_shard${s}"
  num_steps_wait: 10
  num_steps_execute: 8
  num_trials_per_task: 1
  seed: 42
  image_size: 256
  resume_episodes: 0
  resume_successes: 0
YAML
  echo "[envgen] launching $TAG shard$s on GPU$gpu ($(echo $ids | tr ',' '\n' | wc -l) tasks)"
  CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH="$PWD/$PLUS_DIR:$PWD" nohup \
    $VENV scripts/libero_plus_bench_eval.py --config "$cfg" \
    > "logs/plus_envgen_${TAG}_shard${s}.log" 2>&1 &
done
echo "[envgen] all $NSHARD shards launched for $TAG. tail logs/plus_envgen_${TAG}_shard*.log"
