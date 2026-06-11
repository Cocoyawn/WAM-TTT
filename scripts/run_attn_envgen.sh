#!/usr/bin/env bash
# Launch the ATTENTION-baseline env-gen eval (8 shards covering all 1627 tasks) ONLY
# when cgroup RAM has room. The real bottleneck is the 416GB cgroup, NOT GPU. Each
# eval worker ≈ 22GB RSS. We launch in waves: as many workers as RAM allows (capped
# at 8), 2 per GPU, staggered 40s to avoid load-spike OOM. Idempotent + resumable:
# skips shards whose result dir already exists; re-running launches only what's missing.
set -u
cd /mnt/afs-h200/yuyangcheng/workplace/VLANeXt
REPO=$PWD
PLUS_DIR=third_party/LIBERO-plus
VENV=/mnt/afs-h200/yuyangcheng/venvs/fla_triton32/bin/python
RESDIR="$REPO/VLANeXt_ablation_wm/attention_libero_spatial"

export WANDB_API_KEY=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
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa TORCHDYNAMO_DISABLE=1
export LIBERO_CONFIG_PATH=/tmp/libero_plus_cfg

WORKER_GB=22          # measured RSS per eval worker
RESERVE_GB=24         # safety headroom under the cgroup limit
LIM=$(( $(cat /sys/fs/cgroup/memory/memory.limit_in_bytes)/1073741824 ))

free_ram() { echo $(( LIM - $(cat /sys/fs/cgroup/memory/memory.usage_in_bytes)/1073741824 )); }

# which attention shards still need launching (no running proc AND no result dir yet)
pending=()
for s in 0 1 2 3 4 5 6 7; do
  pgrep -f "envgen_attn_shard${s}\.yaml" >/dev/null 2>&1 && continue
  ls -d "$RESDIR"/*envgen_attn_shard${s}* >/dev/null 2>&1 && continue
  pending+=("$s")
done
[ ${#pending[@]} -eq 0 ] && { echo "all attention env-gen shards launched/done, nothing to do"; exit 0; }

avail=$(free_ram)
budget=$(( (avail - RESERVE_GB) / WORKER_GB ))
[ "$budget" -lt 1 ] && { echo "RAM free=${avail}GB (limit ${LIM}) → budget<1 worker, WAIT. pending: ${pending[*]}"; exit 0; }
echo "RAM free=${avail}GB → can launch up to $budget workers. pending shards: ${pending[*]}"

# launch min(budget, #pending) workers, round-robin GPUs 0-3, staggered
launched=0
gpus=(0 1 2 3); gi=0
for s in "${pending[@]}"; do
  [ "$launched" -ge "$budget" ] && break
  gpu=${gpus[$gi]}; gi=$(( (gi+1) % 4 ))
  cfg="config/libero_plus_envgen_attn_shard${s}.yaml"
  CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH="$REPO/$PLUS_DIR:$REPO" \
    nohup bash -c "cd $REPO/$PLUS_DIR && $VENV $REPO/scripts/libero_plus_bench_eval.py --config $REPO/$cfg" \
    >> "logs/plus_envgen_attn_shard${s}.log" 2>&1 &
  echo "launched attn shard$s on GPU$gpu (pid $!)  [free now $(free_ram)GB]"
  launched=$((launched+1))
  sleep 40
done
echo "launched $launched attention worker(s) this wave; re-run to fill remaining as RAM frees."
