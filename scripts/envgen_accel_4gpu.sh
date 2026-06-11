#!/usr/bin/env bash
# Wait until chunk256 base eval frees GPU0/1, then accelerate the TTT-16 env-gen
# eval from 2 GPUs to 4 by launching shard2/shard3 on GPU0/1 covering the tasks
# the running shard0/1 haven't reached yet (the TAIL of their lists). Existing
# shard0/1 on GPU2/3 keep running from the front; any overlap is deduped by
# task_id at aggregation time. Idempotent via .envgen4x_launched marker.
set -u
cd /mnt/afs-h200/yuyangcheng/workplace/VLANeXt
REPO=$PWD
PLUS_DIR=third_party/LIBERO-plus
VENV=/mnt/afs-h200/yuyangcheng/venvs/fla_triton32/bin/python
CKPT="$REPO/VLANeXt_ablation_wm/ttt_libero_spatial/checkpoint_30000.pt"
MARK="$REPO/VLANeXt_ablation_wm/ttt_libero_spatial/.envgen4x_launched"

export WANDB_API_KEY=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
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa TORCHDYNAMO_DISABLE=1
export LIBERO_CONFIG_PATH=/tmp/libero_plus_cfg

[ -f "$MARK" ] && { echo "already launched 4x accel, exit"; exit 0; }

free_gpus() {
  # "available" = can still FIT another ~10GB eval (new GPU-free definition),
  # NOT fully idle. <60GB used means there's room to share-launch an env-gen shard.
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -F',' '$2+0 < 60000 {gsub(/ /,"",$1); print $1}'
}

mapfile -t G < <(free_gpus)
# need GPU0 and GPU1 free (chunk256 eval done)
if [[ ! " ${G[*]} " =~ " 0 " ]] || [[ ! " ${G[*]} " =~ " 1 " ]]; then
  echo "GPU0/1 not available (mem too full to share); free: ${G[*]:-none}. wait."
  exit 0
fi

# Build the set of env-gen tasks the running shard0/1 have NOT reached yet:
# take the LAST portion of each running shard's task list (they run front->back).
$VENV - <<'PY'
import yaml, re, json
# running shards' task lists + how many done
shards={}
for s in (0,1):
    ids=yaml.safe_load(open(f'config/libero_plus_envgen_ttt16_shard{s}.yaml'))['eval']['task_ids']
    t=open(f'logs/plus_envgen_ttt16_shard{s}.log',errors='ignore').read().replace('\r','\n')
    m=re.findall(r'episodes completed so far: (\d+)',t)
    done=int(m[-1]) if m else 0
    shards[s]=(ids,done)
# the running shards will keep covering ids[done:]. To avoid double work, hand the
# NEW shards the BACK HALF of each running shard's remaining list; the running ones
# realistically won't reach there before the new ones finish. Overlap (if any) is
# deduped at aggregation by task_id.
new0=[]; new1=[]
for s,(ids,done) in shards.items():
    rem=ids[done:]
    mid=done+len(rem)//2           # running shard covers [done:mid] front-half before new ones finish
    tail=ids[mid:]                 # new shards take the tail
    (new0 if s==0 else new1).extend(tail)
json.dump(new0, open('/tmp/envgen_shard2_ids.json','w'))
json.dump(new1, open('/tmp/envgen_shard3_ids.json','w'))
print(f"shard2(GPU0): {len(new0)} tasks | shard3(GPU1): {len(new1)} tasks")
PY

for pair in "2:0:/tmp/envgen_shard2_ids.json" "3:1:/tmp/envgen_shard3_ids.json"; do
  tag="${pair%%:*}"; rest="${pair#*:}"; gpu="${rest%%:*}"; idf="${rest##*:}"
  ids=$($VENV -c "import json;print(','.join(map(str,json.load(open('$idf')))))")
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
  echo "launched env-gen shard$tag on GPU$gpu"
done
touch "$MARK"
echo "4x acceleration launched (now shards 0,1 on GPU2/3 + shards 2,3 on GPU0/1)"
