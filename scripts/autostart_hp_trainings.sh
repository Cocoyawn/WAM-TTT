#!/usr/bin/env bash
# Auto-launch the 3 HP cross-suite TTT-256 trainings (object, goal, long) as soon as
# resources free up. Polls every 5 min. Launches one 2-GPU DDP run when >=2 GPUs are
# under 40GB used AND host RAM has >=100GB headroom (a WM training needs ~82GB RSS).
# Runs them sequentially-ish: starts the next only after enough room reappears.
# Idempotent: skips a suite whose checkpoint_final.pt already exists or is already running.
set -u
cd /mnt/afs-h200/yuyangcheng/workplace/VLANeXt
REPO=$PWD
VENV=/mnt/afs-h200/yuyangcheng/venvs/fla_triton32/bin
export WANDB_API_KEY=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
export TORCHDYNAMO_DISABLE=1
export PYTHONPATH=$REPO

SUITES=(object goal long)
PORT=29560

free_gpus() {  # echo indices of GPUs under 40GB used
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | \
    awk -F',' '{u=$2+0; if(u<40000) print $1}'
}
ram_free_gb() {
  lim=$(cat /sys/fs/cgroup/memory/memory.limit_in_bytes)
  use=$(cat /sys/fs/cgroup/memory/memory.usage_in_bytes)
  echo $(( (lim - use) / 1000000000 ))
}
suite_done() {  # 0 if final ckpt exists
  ls VLANeXt_ablation_wm/*ttt_chunk256_libero_$1*/checkpoint_final.pt >/dev/null 2>&1
}
suite_running() {
  ps -eo cmd | grep -F "ablation_wm_ttt_chunk256_libero_$1.yaml" | grep -v grep >/dev/null 2>&1
}

echo "[autostart] watching resources for HP trainings: ${SUITES[*]}"
for s in "${SUITES[@]}"; do
  if suite_done "$s"; then echo "[autostart] $s already done, skip"; continue; fi
  # wait for room
  while true; do
    mapfile -t G < <(free_gpus)
    R=$(ram_free_gb)
    if suite_running "$s"; then echo "[autostart] $s already running, moving on"; break; fi
    if [ "${#G[@]}" -ge 2 ] && [ "$R" -ge 100 ]; then
      g0=${G[0]}; g1=${G[1]}
      log=$REPO/logs/ablation_wm_ttt_chunk256_libero_$s.log
      echo "[autostart] launching $s on GPU$g0,$g1 (RAM ${R}GB free) -> $log"
      CUDA_VISIBLE_DEVICES=$g0,$g1 nohup $VENV/torchrun --nproc_per_node=2 --master_port=$PORT \
        scripts/train.py --config config/ablation_wm_ttt_chunk256_libero_$s.yaml > "$log" 2>&1 &
      PORT=$((PORT+1))
      sleep 300   # let it grab resources before considering the next suite
      break
    fi
    sleep 300
  done
done
echo "[autostart] all 3 HP trainings dispatched (or already done/running)"
