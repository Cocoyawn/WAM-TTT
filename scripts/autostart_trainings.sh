#!/usr/bin/env bash
# Auto-launch queued VLANeXt trainings as GPUs free up. Polls every 5 min.
# QUEUE ORDER (priority): SWA series (swa_gdn, swa_ttt) FIRST, then HP cross-suite
# (object, goal, long). Each is a 2-GPU DDP run.
#
# Launch gate (BOTH must hold — the OOM bug was checking only RAM):
#   * >=2 GPUs each with >=30000 MiB FREE  (not just "<40GB used" — a card shared with
#     env-gen workers can read <40GB used yet have no room for a 25GB/GPU training)
#   * host RAM (cgroup) >=100 GB free  (a WM 2-proc DDP needs ~82GB RSS)
# Idempotent: skips an experiment whose checkpoint_final.pt exists or that's already running.
set -u
cd /mnt/afs-h200/yuyangcheng/workplace/VLANeXt
REPO=$PWD
VENV=/mnt/afs-h200/yuyangcheng/venvs/fla_triton32/bin
export WANDB_API_KEY="$WANDB_API_KEY"
export TORCHDYNAMO_DISABLE=1
export PYTHONPATH=$REPO

# experiment queue: "<tag>:<config-basename>:<result-dir-glob>"  (priority order)
QUEUE=(
  "swa_gdn:ablation_wm_swa_gdn:*swa_gdn*"
  "swa_ttt:ablation_wm_swa_ttt:*swa_ttt*"
  "hp_object:ablation_wm_ttt_chunk256_libero_object:*ttt_chunk256_libero_object*"
  "hp_goal:ablation_wm_ttt_chunk256_libero_goal:*ttt_chunk256_libero_goal*"
  "hp_long:ablation_wm_ttt_chunk256_libero_long:*ttt_chunk256_libero_long*"
)
PORT=29570
MIN_GPU_FREE_MIB=30000
MIN_RAM_FREE_GB=100

# echo indices of GPUs with >= MIN_GPU_FREE_MIB free
free_gpus() {
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | \
    awk -F',' -v m=$MIN_GPU_FREE_MIB '{f=$2+0; if(f>=m) print $1}'
}
ram_free_gb() {
  lim=$(cat /sys/fs/cgroup/memory/memory.limit_in_bytes)
  use=$(cat /sys/fs/cgroup/memory/memory.usage_in_bytes)
  echo $(( (lim - use) / 1000000000 ))
}
exp_done()    { ls VLANeXt_ablation_wm/$1/checkpoint_final.pt >/dev/null 2>&1; }
exp_running() { ps -eo cmd | grep -F "config/$1.yaml" | grep -v grep >/dev/null 2>&1; }

echo "[autostart] queue (priority): $(for e in "${QUEUE[@]}"; do echo -n "${e%%:*} "; done)"
for entry in "${QUEUE[@]}"; do
  tag=${entry%%:*}; rest=${entry#*:}; cfg=${rest%%:*}; glob=${rest#*:}
  if exp_done "$glob"; then echo "[autostart] $tag already has final ckpt, skip"; continue; fi
  while true; do
    if exp_running "$cfg"; then echo "[autostart] $tag already running, next"; break; fi
    mapfile -t G < <(free_gpus); R=$(ram_free_gb)
    if [ "${#G[@]}" -ge 2 ] && [ "$R" -ge "$MIN_RAM_FREE_GB" ]; then
      g0=${G[0]}; g1=${G[1]}
      log=$REPO/logs/$cfg.log
      echo "[autostart] launching $tag on GPU$g0,$g1 (RAM ${R}GB, GPUs free: ${G[*]}) -> $log"
      CUDA_VISIBLE_DEVICES=$g0,$g1 nohup $VENV/torchrun --nproc_per_node=2 --master_port=$PORT \
        scripts/train.py --config config/$cfg.yaml > "$log" 2>&1 &
      PORT=$((PORT+1))
      sleep 420   # let it allocate GPU+RAM before evaluating the next experiment's gate
      break
    fi
    sleep 300
  done
done
echo "[autostart] all queued trainings dispatched (or already done/running)"
