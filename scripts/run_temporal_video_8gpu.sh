#!/usr/bin/env bash
# Temporal open-loop prediction video on an 8-GPU box: shard consecutive episode
# timesteps across GPUs, predict each, stitch into one mp4.
#   GT | VQ-recon | prediction, played at FPS along the episode timeline.
#
# Usage (on the free remote 8-GPU machine):
#   bash scripts/run_temporal_video_8gpu.sh
#   CKPT=... N_FRAMES=96 FPS=10 GPUS=0,1,2,3,4,5,6,7 bash scripts/run_temporal_video_8gpu.sh
set -u
cd /mnt/afs-h200/yuyangcheng/workplace/VLANeXt
REPO=$PWD
VENV=/mnt/afs-h200/yuyangcheng/venvs/fla_triton32/bin/python
CKPT="${CKPT:-$REPO/VLANeXt_robolab_ttt/chunk256_molmoact_droid_long/checkpoint_final.pt}"
N_FRAMES="${N_FRAMES:-0}"          # 0 = full episode (start -> task end)
EPISODE_POS="${EPISODE_POS:-74601}" # which episode (list position); pos74601 len=139
FPS="${FPS:-10}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
OUTDIR="${OUTDIR:-$REPO/docs/tvid_$(basename "$(dirname "$CKPT")")_$(basename "$CKPT" .pt)}"
OUTMP4="${OUTMP4:-${OUTDIR}.mp4}"
mkdir -p "$OUTDIR"

IFS=',' read -ra GA <<< "$GPUS"; NS=${#GA[@]}
echo "[tvid] ckpt=$CKPT  frames=$N_FRAMES  shards=$NS  fps=$FPS  -> $OUTMP4"
pids=()
for k in "${!GA[@]}"; do
  g=${GA[$k]}
  TORCHDYNAMO_DISABLE=1 EVAL_WM_NO_COMPILE=1 PYTHONPATH=$REPO CUDA_VISIBLE_DEVICES=$g \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    $VENV scripts/gen_temporal_pred_video.py --ckpt "$CKPT" --n-frames "$N_FRAMES" \
      --episode-pos "$EPISODE_POS" \
      --shard "$k" --num-shards "$NS" --outdir "$OUTDIR" \
      > "$OUTDIR/shard_${k}.log" 2>&1 &
  pids+=($!)
  echo "  shard $k -> GPU $g (pid $!)"
done
echo "[tvid] waiting for $NS shards ... (live progress below)"
# 解析本 episode 实际帧数作为进度分母(N_FRAMES=0 时从 shard 日志读 length)
TOTAL="$N_FRAMES"
if [ "$TOTAL" -eq 0 ]; then
  for _ in $(seq 1 60); do
    TOTAL=$(grep -oE 'length=[0-9]+' "$OUTDIR/shard_0.log" 2>/dev/null | head -1 | grep -oE '[0-9]+')
    [ -n "$TOTAL" ] && break; TOTAL=0; sleep 5
  done
  [ "$TOTAL" -gt 0 ] || TOTAL=150
  echo "[tvid] full episode length = $TOTAL frames"
fi
# 实时进度条:按实际产出的 frame_*.png 计数(真实工作量)
anyalive(){ for p in "$@"; do kill -0 "$p" 2>/dev/null && return 0; done; return 1; }
bar(){ local d=$1 t=$2; local n=$((d*30/t)); [ $n -gt 30 ] && n=30; local i; printf '['; for ((i=0;i<30;i++)); do [ $i -lt $n ] && printf '#' || printf '-'; done; printf ']'; }
start=$(date +%s)
while anyalive "${pids[@]}"; do
  done=$(ls "$OUTDIR"/frame_*.png 2>/dev/null | wc -l)
  el=$(( $(date +%s) - start ))
  printf '\r[tvid] %s %d/%d frames (%d%%)  elapsed %dm%02ds   ' "$(bar "$done" "$TOTAL")" "$done" "$TOTAL" "$((done*100/TOTAL))" "$((el/60))" "$((el%60))"
  sleep 5
done
for p in "${pids[@]}"; do wait "$p"; done
printf '\n'

NF=$(ls "$OUTDIR"/frame_*.png 2>/dev/null | wc -l)
echo "[tvid] $NF/$TOTAL frames generated"
[ "$NF" -gt 0 ] || { echo "[tvid] ERROR: no frames (check $OUTDIR/shard_*.log)"; exit 1; }
ffmpeg -y -framerate "$FPS" -pattern_type glob -i "$OUTDIR/frame_*.png" \
  -c:v libx264 -pix_fmt yuv420p -movflags +faststart "$OUTMP4" 2>&1 | tail -2
echo "[tvid] DONE -> $OUTMP4  (cols: GT | VQ-recon | prediction)"
