#!/usr/bin/env bash
# Batch: for each episode, render TTT (GPUs 4,5) + attn (GPUs 6,7) in parallel over the
# FULL episode, then compose a GT|TTT|attn 3-col mp4. Churns through all episodes.
# Usage: bash scripts/run_videos_batch.sh "74597,74591,..."  (defaults to the 10 below)
set -u
cd /mnt/afs-h200/yuyangcheng/workplace/VLANeXt
REPO=$PWD
V=/mnt/afs-h200/yuyangcheng/venvs/fla_triton32/bin/python
TTT_CKPT=$REPO/VLANeXt_robolab_ttt/chunk256_molmoact_droid_long/checkpoint_final.pt
ATT_CKPT=$REPO/VLANeXt_robolab_attn/molmoact_droid_cont/checkpoint_final.pt
TTT_GPUS="${TTT_GPUS:-4,5}"
ATT_GPUS="${ATT_GPUS:-6,7}"
FPS="${FPS:-12}"
POS_LIST="${1:-74597,74591,74590,74588,74586,74584,74583,74580,74578,74577}"

IFS=',' read -ra EPS <<< "$POS_LIST"
echo "[batch] ${#EPS[@]} episodes; TTT on $TTT_GPUS, attn on $ATT_GPUS"
for ep in "${EPS[@]}"; do
  tdir=$REPO/docs/tvid_ttt_ep${ep}; adir=$REPO/docs/tvid_attn_ep${ep}
  out=$REPO/docs/robolab_ep${ep}_GT_TTT_attn.mp4
  if [ -f "$out" ]; then echo "[batch] ep$ep already done -> skip"; continue; fi
  echo "[batch] === ep$ep: rendering TTT + attn (full episode) ==="
  GPUS=$TTT_GPUS EPISODE_POS=$ep FPS=$FPS MASTER_PORT=29661 \
    CKPT=$TTT_CKPT OUTDIR=$tdir OUTMP4=$tdir.mp4 \
    bash scripts/run_temporal_video_8gpu.sh > $REPO/logs/vbatch_ttt_ep${ep}.log 2>&1 &
  p1=$!
  GPUS=$ATT_GPUS EPISODE_POS=$ep FPS=$FPS MASTER_PORT=29662 \
    CKPT=$ATT_CKPT OUTDIR=$adir OUTMP4=$adir.mp4 \
    bash scripts/run_temporal_video_8gpu.sh > $REPO/logs/vbatch_attn_ep${ep}.log 2>&1 &
  p2=$!
  wait $p1; wait $p2
  echo "[batch] ep$ep: composing 3-col comparison ..."
  $V scripts/compose_temporal_cmp.py "$tdir" "$adir" "$out" "$FPS" \
    && rm -rf "$tdir" "$adir" "$tdir.mp4" "$adir.mp4" \
    && echo "[batch] ep$ep DONE -> $out"
done
echo "[batch] ALL DONE. comparison videos:"
ls -la $REPO/docs/robolab_ep*_GT_TTT_attn.mp4 2>/dev/null | awk '{print $5,$9}'
