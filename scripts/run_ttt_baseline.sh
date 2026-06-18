#!/usr/bin/env bash
# ============================================================================
# TTT CUDA-kernel BASELINE + TIMING — run on a CLEAN, EXCLUSIVE GPU.
# Self-contained: builds the JIT extension (auto-detects sm arch), runs the
# three benches, writes docs/ttt_kernel_baseline.md.
#
# Usage on the new machine:
#   1) edit REPO / VENV below if paths differ
#   2) pick a free GPU index (nvidia-smi); pass it as $1 (default 0)
#   3) bash run_ttt_baseline.sh 0
#
# REQUIREMENTS on the new machine:
#   - the shared venv (torch 2.6+cu124, triton) OR an equivalent venv
#   - nvcc matching the torch CUDA version (12.4); CUDA_HOME set
#   - the Qwen3-VL model at the path build() expects (for bench_ttt_vs_attn /
#     bench_chunk_infer_latency). If absent, the layer-profile bench still runs.
# ============================================================================
set -u

REPO=/mnt/afs-h200/yuyangcheng/workplace/VLANeXt
VENV=/mnt/afs-h200/yuyangcheng/venvs/fla_triton32/bin/python
GPU="${1:-0}"

cd "$REPO" || { echo "repo not found: $REPO"; exit 1; }
export PYTHONPATH="$REPO"
export CUDA_VISIBLE_DEVICES="$GPU"
unset TORCH_CUDA_ARCH_LIST            # let the extension auto-detect (sm_80/sm_90/...)
export TTT_CUDA_VERBOSE=1

OUT="$REPO/docs/ttt_kernel_baseline.md"
mkdir -p "$REPO/docs"
ts() { TZ='UTC-8' date '+%F %H:%M:%S'; }

echo "# TTT CUDA-kernel baseline (GPU $GPU)"            >  "$OUT"
echo ""                                                  >> "$OUT"
echo "Generated: $(ts) 北京时间"                          >> "$OUT"
$VENV -c "import torch; print('- device:', torch.cuda.get_device_name(0)); print('- capability:', torch.cuda.get_device_capability(0)); print('- torch:', torch.__version__)" 2>/dev/null | sed 's/^/  /' >> "$OUT"
echo ""                                                  >> "$OUT"

# ---------- guard: is the chosen GPU actually clean? ----------
read used util < <($VENV - <<'PY'
import torch, subprocess, os
idx = 0  # CUDA_VISIBLE_DEVICES already remaps so device 0 is the chosen one
free, total = torch.cuda.mem_get_info(0)
used_mb = (total - free) / 1e6
print(int(used_mb), 0)
PY
)
echo "[info] $(ts) GPU$GPU used≈${used} MiB (before our run)"
if [ "${used:-9999}" -gt 2000 ]; then
  echo "[WARN] GPU$GPU already has >2GB used — timings may be POLLUTED by other tenants." | tee -a "$OUT"
fi

# ======================================================================
# Bench 1: training step time — attention vs torch-TTT vs CUDA-TTT
#   three compile settings: OFF / ON
# ======================================================================
echo "## 1. Training step time (bench_ttt_vs_attn)"      >> "$OUT"
echo '```'                                               >> "$OUT"
for DYN in 1 0; do
  lbl=$([ "$DYN" = 1 ] && echo "compile OFF" || echo "compile ON")
  echo "[run] $(ts) bench_ttt_vs_attn — $lbl"
  echo "### compile $([ "$DYN" = 1 ] && echo OFF || echo ON)" >> "$OUT"
  TORCHDYNAMO_DISABLE=$DYN $VENV -m scripts.bench_ttt_vs_attn 2>&1 \
    | grep -aE "batch=|attention|torch-TTT|CUDA-TTT" | tee -a "$OUT"
done
echo '```'                                               >> "$OUT"
echo ""                                                  >> "$OUT"

# ======================================================================
# Bench 2: per-layer kernel profile (torch vs CUDA) — does NOT need Qwen
# ======================================================================
echo "## 2. Per-layer kernel profile (bench_ttt_layer_profile)" >> "$OUT"
echo '```'                                               >> "$OUT"
echo "[run] $(ts) bench_ttt_layer_profile"
TORCHDYNAMO_DISABLE=1 $VENV -m scripts.bench_ttt_layer_profile 2>&1 \
  | grep -avE "FutureWarning|warnings.warn|recommended|below the" | tee -a "$OUT"
echo '```'                                               >> "$OUT"
echo ""                                                  >> "$OUT"

# ======================================================================
# Bench 3: deployment inference latency — attn vs torch-TTT vs CUDA-TTT sweep
# ======================================================================
echo "## 3. Inference latency (bench_chunk_infer_latency)" >> "$OUT"
echo '```'                                               >> "$OUT"
echo "[run] $(ts) bench_chunk_infer_latency"
TORCHDYNAMO_DISABLE=1 $VENV -m scripts.bench_chunk_infer_latency 2>&1 \
  | grep -aE "attention|torch-TTT|CUDA-TTT|baseline|predict_action|iters" | tee -a "$OUT"
echo '```'                                               >> "$OUT"

echo ""
echo "==== DONE $(ts) 北京 — results in $OUT ===="
