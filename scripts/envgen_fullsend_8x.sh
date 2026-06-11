#!/usr/bin/env bash
# FULL-SEND env-gen: the 4 running shards (shard0..3) are CPU-render bound, each
# using only ~1 core + ~10GB GPU. Huge idle headroom (99 cores, 30GB/card free).
# Launch 8 MORE eval workers (2 per GPU → 12-way total) to chew the UNREACHED TAILS
# of the running shards. Running shards march front->back; we hand new workers the
# back 2/3 of each remaining tail, so overlap is minimal and deduped by task name
# at aggregation. Idempotent via .fullsend8x_launched marker.
set -u
cd /mnt/afs-h200/yuyangcheng/workplace/VLANeXt
REPO=$PWD
PLUS_DIR=third_party/LIBERO-plus
VENV=/mnt/afs-h200/yuyangcheng/venvs/fla_triton32/bin/python
CKPT="$REPO/VLANeXt_ablation_wm/ttt_libero_spatial/checkpoint_30000.pt"
MARK="$REPO/VLANeXt_ablation_wm/ttt_libero_spatial/.fullsend8x_launched"

export WANDB_API_KEY=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
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa TORCHDYNAMO_DISABLE=1
export LIBERO_CONFIG_PATH=/tmp/libero_plus_cfg

[ -f "$MARK" ] && { echo "fullsend 8x already launched, exit"; exit 0; }

# safety: ensure each card has >=18GB free before adding 2x10GB workers (→ <80GB).
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F',' '{
  used=$2+0; if (80000-used < 22000) { print "GPU"$1" only "(80000-used)"MB free, abort"; exit 1 } }' || {
  echo "insufficient GPU headroom, abort"; exit 1; }

# Build 8 new task-id lists from the unreached tails of the 4 running shards.
$VENV - <<'PY'
import yaml, os, json
base="VLANeXt_ablation_wm/ttt_libero_spatial"
pool=[]
for s in range(4):
    ids=yaml.safe_load(open(f'config/libero_plus_envgen_ttt16_shard{s}.yaml'))['eval']['task_ids']
    d=f"{base}/libero_spatial_checkpoint_30000_exec8_diff5_libero_plus_envgen_ttt16_shard{s}"
    done=len(os.listdir(d)) if os.path.isdir(d) else 0
    rem=ids[done:]
    # running shard will cover front 1/3 of remaining before new workers finish;
    # hand new workers the back 2/3.
    cut=len(rem)//3
    pool.extend(rem[cut:])
# dedup pool preserving order
seen=set(); pool=[x for x in pool if not (x in seen or seen.add(x))]
# split into 8 contiguous chunks
N=8; k=(len(pool)+N-1)//N
for i in range(N):
    json.dump(pool[i*k:(i+1)*k], open(f'/tmp/envgen_fs_shard{4+i}_ids.json','w'))
print("pool size:", len(pool), "→ 8 workers of ~", k, "tasks each")
PY

# GPU assignment: 2 new workers per card → tags 4..11 on GPUs 0,0,1,1,2,2,3,3
i=0
for gpu in 0 0 1 1 2 2 3 3; do
  tag=$((4+i)); i=$((i+1))
  idf="/tmp/envgen_fs_shard${tag}_ids.json"
  ids=$($VENV -c "import json;print(','.join(map(str,json.load(open('$idf')))))")
  [ -z "$ids" ] && { echo "shard$tag empty, skip"; continue; }
  cfg="config/libero_plus_envgen_ttt16_shard${tag}.yaml"
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
  shard_tag: "envgen_ttt16_shard${tag}"
  num_steps_wait: 10
  num_steps_execute: 8
  num_trials_per_task: 1
  seed: 42
  image_size: 256
  resume_episodes: 0
  resume_successes: 0
YAML
  CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH="$REPO/$PLUS_DIR:$REPO" \
    nohup bash -c "cd $REPO/$PLUS_DIR && $VENV $REPO/scripts/libero_plus_bench_eval.py --config $REPO/$cfg" \
    > "logs/plus_envgen_ttt16_shard${tag}.log" 2>&1 &
  echo "launched env-gen worker shard$tag on GPU$gpu (pid $!)"
  sleep 3
done
touch "$MARK"
echo "8x full-send launched (12-way env-gen total: shards 0-3 + new 4-11)"
