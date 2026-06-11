#!/bin/bash
# World-model video-quality eval: TTT-16 vs Attention (30k final), shared cache.
# Runs loss(cheap)+gen(AR) for both ckpts. TTT on GPU2, Attn on GPU3, in parallel.
# Assumes cache already built (scripts/cache_wm_eval_samples.py).
set -e
cd "$(dirname "$0")/.."

VENV=/mnt/afs-h200/yuyangcheng/venvs/fla_triton32/bin/python
TTT_CKPT=VLANeXt_ablation_wm/ttt_libero_spatial/checkpoint_final.pt
ATTN_CKPT=VLANeXt_ablation_wm/attention_libero_spatial/checkpoint_final.pt
SUITES="libero_spatial_no_noops libero_object_no_noops libero_goal_no_noops"
GEN_LIMIT="${GEN_LIMIT:-32}"     # AR gen is ~3s/sample; cap per suite
LOSS_LIMIT="${LOSS_LIMIT:-64}"   # teacher-forced loss is cheap; use full
mkdir -p logs

# LPIPS needs proxy for its first weight download (cached after).
source /mnt/afs-h200/yuyangcheng/workplace/proxy.sh 2>/dev/null || true
export HF_HUB_DISABLE_PROGRESS_BARS=1

run_model () {  # $1=gpu $2=tag $3=ckpt
  local gpu=$1 tag=$2 ckpt=$3
  echo "[run] $tag on GPU$gpu"
  CUDA_VISIBLE_DEVICES=$gpu $VENV scripts/eval_wm_quality.py loss \
    --ckpt "$ckpt" --tag "$tag" --device cuda:0 --limit "$LOSS_LIMIT" --suites $SUITES \
    > "logs/wm_${tag}_loss.log" 2>&1
  CUDA_VISIBLE_DEVICES=$gpu $VENV scripts/eval_wm_quality.py gen \
    --ckpt "$ckpt" --tag "$tag" --device cuda:0 --limit "$GEN_LIMIT" --suites $SUITES \
    > "logs/wm_${tag}_gen.log" 2>&1
  echo "[done] $tag"
}

run_model 2 ttt  "$TTT_CKPT"  &
run_model 3 attn "$ATTN_CKPT" &
wait

echo "[summary] aggregating ..."
$VENV scripts/summarize_wm_quality.py --ttt_tag ttt --attn_tag attn
echo "[all done] see VLANeXt_ablation_wm/wm_eval_cache/WM_QUALITY_SUMMARY.md"
