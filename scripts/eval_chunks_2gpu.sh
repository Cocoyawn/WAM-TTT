#!/usr/bin/env bash
# Evaluate chunk64 and chunk256 final checkpoints on the BASE LIBERO-spatial suite
# (10 tasks x 50 trials), 2-shard on GPU0/1 (run in parallel with the GPU2/3
# env-gen eval). Sequential per checkpoint: chunk64 first, then chunk256.
# Aggregates per-task SR into EVAL_SUMMARY.txt in each run dir.
set -u
cd /mnt/afs-h200/yuyangcheng/workplace/VLANeXt
REPO=/mnt/afs-h200/yuyangcheng/workplace/VLANeXt
VENV=/mnt/afs-h200/yuyangcheng/venvs/fla_triton32/bin/python
export WANDB_API_KEY="$WANDB_API_KEY"
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa TORCHDYNAMO_DISABLE=1
export PYTHONPATH="$PWD"

RUNS=(
  "chunk64:VLANeXt_ablation_wm/ttt_chunk64_libero_spatial/checkpoint_final.pt"
  "chunk256:VLANeXt_ablation_wm/ttt_chunk256_libero_spatial/checkpoint_final.pt"
)
# 2 shards: GPU0 = tasks 0-4, GPU1 = tasks 5-9
declare -A SHARD_TASKS=( [0]="0 1 2 3 4" [1]="5 6 7 8 9" )
declare -A SHARD_GPU=( [0]=0 [1]=1 )

for entry in "${RUNS[@]}"; do
  tag="${entry%%:*}"; ckpt="$PWD/${entry##*:}"
  [ -f "$ckpt" ] || { echo "[$tag] no ckpt, skip"; continue; }
  dir="$(dirname "$ckpt")"
  [ -f "$dir/.baseeval_done" ] && { echo "[$tag] base eval already done, skip"; continue; }
  echo "[$(date +%H:%M)] [$tag] launching 2-shard base-spatial eval on GPU0/1"
  for s in 0 1; do
    ids=$(echo ${SHARD_TASKS[$s]} | tr ' ' ',')
    cfg="config/eval_${tag}_base_shard${s}.yaml"
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
  finetuned_checkpoint: "$ckpt"
  task_suite_name: "libero_spatial"
  task_ids: [$ids]
  shard_tag: "base_shard${s}"
  num_steps_wait: 10
  num_steps_execute: 8
  num_trials_per_task: 50
  seed: 42
  image_size: 256
  resume_episodes: 0
  resume_successes: 0
YAML
    gpu=${SHARD_GPU[$s]}
    # base LIBERO import requires cwd=third_party/LIBERO (its libero/ subdir on path)
    CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH="$REPO:$REPO/third_party/LIBERO" \
      nohup bash -c "cd $REPO/third_party/LIBERO && $VENV $REPO/scripts/libero_bench_eval.py --config $REPO/$cfg" \
      > "logs/eval_${tag}_base_shard${s}.log" 2>&1 &
  done
  wait
  touch "$dir/.baseeval_done"
  echo "[$(date +%H:%M)] [$tag] both shards done -> aggregating"
  $VENV - "$dir" "$tag" <<'PY'
import json, glob, sys, os
dir_, tag = sys.argv[1], sys.argv[2]
rows={}
for f in glob.glob(dir_+"/**/task_*_result.json", recursive=True):
    try:
        d=json.load(open(f)); rows[d['task_id']]=d
    except Exception: pass
out=[f"=== {tag} LIBERO-spatial base per-task SR ==="]
ts=te=0
for tid in sorted(rows):
    d=rows[tid]; ts+=d['successes']; te+=d['episodes']
    out.append(f"task{tid}: {d['successes']}/{d['episodes']} = {d['success_rate']*100:.0f}%")
out.append(f"TOTAL: {ts}/{te} = {ts/te*100:.2f}%" if te else "no data")
txt="\n".join(out); print(txt)
open(os.path.join(dir_,"EVAL_SUMMARY.txt"),"w").write(txt+"\n")
PY
done
echo "[$(date +%H:%M)] chunk base-eval all done"
