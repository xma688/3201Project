#!/bin/bash

set -euo pipefail

SCENE_DIR="${1:-}"
OUTPUT_TAG="${2:-}"

if [ -z "$SCENE_DIR" ] || [ -z "$OUTPUT_TAG" ]; then
  echo "Usage: bash part3/scripts/train_part3_3dgs.sh /path/to/your/hybrid_scene OUTPUT_TAG [ITERATIONS]"
  exit 1
fi
SCENE_DIR="$(cd "$SCENE_DIR" && pwd)"

ITERATIONS=""
if [ $# -ge 3 ] && [[ "${3:-}" =~ ^[0-9]+$ ]]; then
  ITERATIONS="${3:-}"
  shift 3
else
  shift 2
fi
EXTRA_ARGS=("$@")
CALL_CWD="$(pwd)"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PART3_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$PART3_ROOT/.." && pwd)"
REPO="$PROJECT_ROOT/gaussian-splatting"
TRAIN_APP="$PART3_ROOT/apps/train_3dgs_confidence.py"
WORKSPACE_ROOT="${PART3_WORKSPACE_ROOT:-$PART3_ROOT/workspace}"
OUT_DIR="${PART3_3DGS_OUTPUT_ROOT:-$WORKSPACE_ROOT/3dgs_outputs}/$OUTPUT_TAG"
LOG_DIR="$OUT_DIR/logs"
mkdir -p "$LOG_DIR"

normalize_extra_args() {
  local normalized=()
  local idx=0
  while [ "$idx" -lt "${#EXTRA_ARGS[@]}" ]; do
    local arg="${EXTRA_ARGS[$idx]}"
    normalized+=("$arg")
    if [ "$arg" = "--confidence_manifest" ] && [ $((idx + 1)) -lt "${#EXTRA_ARGS[@]}" ]; then
      idx=$((idx + 1))
      local value="${EXTRA_ARGS[$idx]}"
      if [[ "$value" != /* ]]; then
        value="$CALL_CWD/$value"
      fi
      normalized+=("$value")
    fi
    idx=$((idx + 1))
  done
  EXTRA_ARGS=("${normalized[@]}")
}

normalize_extra_args

cd "$REPO"

CMD=(
  python3 "$TRAIN_APP"
  -s "$SCENE_DIR"
  -m "$OUT_DIR"
  --eval
  --disable_viewer
  --confidence_from_alpha
  --test_iterations 1000 3000 7000 15000 30000
  --save_iterations 1000 3000 7000 15000 30000
  --checkpoint_iterations 15000 30000
)

if [ -n "$ITERATIONS" ]; then
  CMD+=(--iterations "$ITERATIONS")
fi

if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
  CMD+=("${EXTRA_ARGS[@]}")
fi

"${CMD[@]}" 2>&1 | tee "$LOG_DIR/train_console.log"
