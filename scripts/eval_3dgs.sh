#!/bin/bash
# 用法:
# bash scripts/eval_3dgs.sh DL3DV-2 PlanA
# bash scripts/eval_3dgs.sh Re10k-1 PlanB
# bash scripts/eval_3dgs.sh 405841_FRONT PlanA
# bash scripts/eval_3dgs.sh 405841/FRONT PlanA
# bash scripts/eval_3dgs.sh 405841 PlanA

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO="$ROOT/gaussian-splatting"

SCENE_IN="${1:-DL3DV-2}"
PLAN="${2:-PlanA}"

case "$PLAN" in
  PlanA|PlanB)
    ;;
  *)
    echo "Unsupported plan: $PLAN"
    echo "Supported plans: PlanA | PlanB"
    exit 1
    ;;
esac

# 规范化 scene 名称
# SCENE_TAG: 对应 outputs/3dgs/$PLAN/$SCENE_TAG
case "$SCENE_IN" in
  DL3DV-2)
    SCENE_TAG="DL3DV-2"
    ;;
  Re10k-1)
    SCENE_TAG="Re10k-1"
    ;;
  405841_FRONT|405841/FRONT|405841)
    SCENE_TAG="405841_FRONT"
    ;;
  *)
    echo "Unsupported scene: $SCENE_IN"
    echo "Supported scenes:"
    echo "  DL3DV-2"
    echo "  Re10k-1"
    echo "  405841_FRONT"
    echo "  405841/FRONT"
    echo "  405841"
    exit 1
    ;;
esac

OUT_DIR="$ROOT/outputs/3dgs/$PLAN/$SCENE_TAG"
LOG_DIR="$OUT_DIR/logs"

mkdir -p "$LOG_DIR"

if [ ! -d "$OUT_DIR" ]; then
  echo "Error: trained model folder not found: $OUT_DIR"
  exit 1
fi

cd "$REPO"

python render.py \
  -m "$OUT_DIR" \
  --skip_train \
  2>&1 | tee "$LOG_DIR/render_test_console.log"

python metrics.py \
  -m "$OUT_DIR" \
  2>&1 | tee "$LOG_DIR/metrics_test_console.log"

{
  echo "========================================"
  echo "SCENE_IN       : $SCENE_IN"
  echo "SCENE_TAG      : $SCENE_TAG"
  echo "PLAN           : $PLAN"
  echo "MODEL          : $OUT_DIR"
  echo "NOTE           : render.py ran with --skip_train, so only test split was rendered."
  echo "========================================"
} | tee "$LOG_DIR/eval_meta.txt"