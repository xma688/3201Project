#!/bin/bash

set -euo pipefail

MODEL_DIR="${1:-}"
if [ -z "$MODEL_DIR" ]; then
  echo "Usage: bash part3/scripts/eval_part3_3dgs.sh /path/to/your/model_dir [ITERATION]"
  exit 1
fi
MODEL_DIR="$(cd "$MODEL_DIR" && pwd)"
ITERATION="${2:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PART3_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$PART3_ROOT/.." && pwd)"
REPO="$PROJECT_ROOT/gaussian-splatting"
LOG_DIR="$MODEL_DIR/logs"
mkdir -p "$LOG_DIR"

cd "$REPO"

RENDER_ARGS=(-m "$MODEL_DIR")
LOG_SUFFIX=""
if [ -n "$ITERATION" ]; then
  RENDER_ARGS+=(--iteration "$ITERATION")
  LOG_SUFFIX="_${ITERATION}"
fi
if [ "${SKIP_TRAIN_RENDER:-0}" = "1" ]; then
  RENDER_ARGS+=(--skip_train)
fi

python3 render.py "${RENDER_ARGS[@]}" 2>&1 | tee "$LOG_DIR/render_test_console${LOG_SUFFIX}.log"
python3 metrics.py -m "$MODEL_DIR" 2>&1 | tee "$LOG_DIR/metrics_test_console${LOG_SUFFIX}.log"
