#!/usr/bin/env bash
# Run the GatedDeltaNet (fla) smoke test inside the isolated venv that provides
# triton>=3.2 (required by the fla triton kernels), without touching the system
# environment. The venv inherits the system torch/CUDA via --system-site-packages
# and only shadows triton.
#
# Usage:
#   bash src/models/run_fla_smoke.sh
set -euo pipefail

VENV="${FLA_VENV:-/mnt/afs-h200/yuyangcheng/venvs/fla_triton32}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if [[ ! -x "$VENV/bin/python" ]]; then
    echo "venv not found at $VENV"
    echo "create it with:"
    echo "  uv venv --system-site-packages --python /usr/bin/python3 $VENV"
    echo "  $VENV/bin/python -m pip install --no-deps triton==3.2.0"
    exit 1
fi

cd "$REPO_ROOT"
# Disable dynamo/inductor: the custom torch 2.3 inductor is incompatible with
# triton 3.2; fla's @torch.compile helpers fall back to eager (numerically same).
export TORCHDYNAMO_DISABLE=1
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

exec "$VENV/bin/python" -m src.models.fla.test_gated_deltanet
