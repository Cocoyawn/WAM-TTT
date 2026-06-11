#!/usr/bin/env bash
# Auto-eval driver for the chunk-size ablation runs.
# Triggered by cron. For each finished training (checkpoint_final.pt present and
# not yet evaluated), runs a 4-GPU sharded LIBERO-spatial eval (OSMesa render),
# then aggregates a per-task success-rate table. Idempotent: a .evaldone marker
# prevents re-evaluating; a .evalrunning marker prevents concurrent launches.
set -u
cd /mnt/afs-h200/yuyangcheng/workplace/VLANeXt

VENV=/mnt/afs-h200/yuyangcheng/venvs/fla_triton32/bin/python
export WANDB_API_KEY=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
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export TORCHDYNAMO_DISABLE=1
export PYTHONPATH="$PWD"

# runs: "tag:save_dir"
RUNS=(
  "chunk64:VLANeXt_ablation_wm/ttt_chunk64_libero_spatial"
  "chunk256:VLANeXt_ablation_wm/ttt_chunk256_libero_spatial"
)

# shard -> task_ids (full 10-task coverage, 4 shards on 4 GPUs)
declare -A SHARD_TASKS=( [0]="0 1 2" [1]="3 4 5" [2]="6 7" [3]="8 9" )

free_gpus() {
  # echo indices of GPUs with <2GB used
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -F',' '$2+0 < 2000 {gsub(/ /,"",$1); print $1}'
}

for entry in "${RUNS[@]}"; do
  tag="${entry%%:*}"; dir="${entry##*:}"
  ckpt="$dir/checkpoint_final.pt"
  [ -f "$ckpt" ] || { echo "[$(date +%H:%M)] $tag: no final ckpt yet, skip"; continue; }
  [ -f "$dir/.evaldone" ] && continue
  [ -f "$dir/.evalrunning" ] && { echo "[$(date +%H:%M)] $tag: eval already running, skip"; continue; }

  mapfile -t GPUS < <(free_gpus)
  if [ "${#GPUS[@]}" -lt 4 ]; then
    echo "[$(date +%H:%M)] $tag: only ${#GPUS[@]} free GPUs (<4), wait for next tick"
    continue
  fi
  touch "$dir/.evalrunning"
  echo "[$(date +%H:%M)] $tag: launching 4-shard eval on GPUs ${GPUS[*]:0:4}"

  for s in 0 1 2 3; do
    cfg="config/eval_${tag}_shard${s}.yaml"
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
  finetuned_checkpoint: "$PWD/$ckpt"
  task_suite_name: "libero_spatial"
  task_ids: [$(echo ${SHARD_TASKS[$s]} | tr ' ' ',')]
  shard_tag: "shard${s}"
  num_steps_wait: 10
  num_steps_execute: 8
  num_trials_per_task: 50
  seed: 42
  image_size: 256
  resume_episodes: 0
  resume_successes: 0
YAML
    gpu=${GPUS[$s]}
    CUDA_VISIBLE_DEVICES=$gpu nohup $VENV scripts/libero_bench_eval.py \
      --config "$cfg" > "logs/eval_${tag}_shard${s}.log" 2>&1 &
  done
  wait
  rm -f "$dir/.evalrunning"
  touch "$dir/.evaldone"
  echo "[$(date +%H:%M)] $tag: all shards done -> aggregating"

  # aggregate per-task table
  $VENV - "$dir" "$tag" <<'PY'
import json, glob, sys, os
dir_, tag = sys.argv[1], sys.argv[2]
rows={}
for f in glob.glob(dir_+"/**/task_*_result.json", recursive=True):
    try:
        d=json.load(open(f)); rows[d['task_id']]=d
    except Exception: pass
out=[f"=== {tag} LIBERO-spatial per-task SR ==="]
ts=te=0
for tid in sorted(rows):
    d=rows[tid]; ts+=d['successes']; te+=d['episodes']
    out.append(f"task{tid}: {d['successes']}/{d['episodes']} = {d['success_rate']*100:.0f}%")
out.append(f"TOTAL: {ts}/{te} = {ts/te*100:.2f}%" if te else "no data")
txt="\n".join(out)
print(txt)
open(os.path.join(dir_, "EVAL_SUMMARY.txt"),"w").write(txt+"\n")
PY
done
echo "[$(date +%H:%M)] auto_eval tick complete"
