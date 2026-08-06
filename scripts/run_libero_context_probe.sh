#!/usr/bin/env bash
# Small paired LIBERO-plus context probe.
# Conditions: no context, same-task/different-instance context, other-task context.
# This script does not disable or toggle TTT updates; all conditions use the
# same checkpoint and the same simulator task IDs.
set -u

REPO="${REPO:-/mnt/afs-h200/yuyangcheng/workplace/VLANeXt}"
PLUS_DIR="${PLUS_DIR:-$REPO/third_party/LIBERO-plus}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/afs-h200/yuyangcheng/venvs/fla_triton32/bin/python}"
DATA_ROOT="${DATA_ROOT:-/mnt/afs-h200/NTU_slab/draven/data/LIBERO_modified}"
TASK_SUITE="${TASK_SUITE:-libero_spatial}"
TASK_IDS="${TASK_IDS:-0,1,2,3}"
NUM_TRIALS="${NUM_TRIALS:-1}"
CONTEXT_FRAMES="${CONTEXT_FRAMES:-2}"
CONTEXT_SEED="${CONTEXT_SEED:-42}"
BANK_ROOT="${BANK_ROOT:-$REPO/context_banks}"
BANK_DIR="$BANK_ROOT/$TASK_SUITE"
BANK_MANIFEST="$BANK_DIR/manifest.json"
CKPT="${CKPT:-$REPO/VLANeXt_ablation_wm/ttt_chunk256_libero_mixed_clean/checkpoint_final.pt}"
TAG="${TAG:-context_probe}"
LOG_DIR="${LOG_DIR:-$REPO/logs}"

if [ ! -x "$PYTHON_BIN" ]; then
  echo "[error] PYTHON_BIN is not executable: $PYTHON_BIN" >&2
  exit 2
fi
if [ ! -f "$CKPT" ]; then
  echo "[error] CKPT does not exist: $CKPT" >&2
  echo "        Set CKPT=/absolute/path/to/checkpoint_final.pt" >&2
  exit 2
fi
if [ ! -d "$PLUS_DIR" ]; then
  echo "[error] LIBERO-plus directory does not exist: $PLUS_DIR" >&2
  exit 2
fi

export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-/tmp/libero_plus_cfg}"
case "$PLUS_DIR" in
  /*) PLUS_PYTHONPATH="$PLUS_DIR" ;;
  *)  PLUS_PYTHONPATH="$REPO/$PLUS_DIR" ;;
esac
export PYTHONPATH="$PLUS_PYTHONPATH:$REPO${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$LOG_DIR" "$BANK_ROOT"

if [ ! -f "$BANK_MANIFEST" ]; then
  echo "[bank] building $BANK_MANIFEST"
  "$PYTHON_BIN" "$REPO/scripts/build_libero_context_bank.py" \
    --data-root "$DATA_ROOT" \
    --suite "$TASK_SUITE" \
    --frames "$CONTEXT_FRAMES" \
    --max-per-instruction 2 \
    --out-dir "$BANK_DIR"
else
  echo "[bank] reusing $BANK_MANIFEST"
fi

TASK_IDS_YAML=$("$PYTHON_BIN" -c 'import sys; print(",".join(str(int(x)) for x in sys.argv[1].split(",") if x.strip()))' "$TASK_IDS")
echo "[probe] ckpt=$CKPT suite=$TASK_SUITE tasks=[$TASK_IDS_YAML] trials=$NUM_TRIALS"

for MODE in none same_task other_task; do
  CFG="/tmp/vlanext_${TAG}_${MODE}.yaml"
  SHARD_TAG="${TAG}_${MODE}"
  cat > "$CFG" <<YAML
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
  task_suite_name: "$TASK_SUITE"
  task_ids: [$TASK_IDS_YAML]
  shard_tag: "$SHARD_TAG"
  num_steps_wait: 10
  num_steps_execute: 8
  num_trials_per_task: $NUM_TRIALS
  seed: 42
  image_size: 256
  resume_episodes: 0
  resume_successes: 0
  context_mode: "$MODE"
  context_frames: $CONTEXT_FRAMES
  context_seed: $CONTEXT_SEED
  context_manifest: "$BANK_MANIFEST"
YAML

  echo "[probe] running mode=$MODE"
  (
    cd "$PLUS_DIR" || exit 3
    "$PYTHON_BIN" "$REPO/scripts/libero_plus_bench_eval.py" --config "$CFG"
  ) 2>&1 | tee "$LOG_DIR/libero_context_${TAG}_${MODE}.log"
  status=${PIPESTATUS[0]}
  if [ "$status" -ne 0 ]; then
    echo "[error] mode=$MODE failed with status=$status" >&2
    exit "$status"
  fi
done

echo "[probe] complete. Results are under the checkpoint directory with tags ${TAG}_{none,same_task,other_task}."
