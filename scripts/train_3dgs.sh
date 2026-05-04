#!/bin/bash
# 用法:
# bash scripts/train_3dgs.sh DL3DV-2 PlanA
# bash scripts/train_3dgs.sh Re10k-1 PlanB
# bash scripts/train_3dgs.sh 405841_FRONT PlanA
# bash scripts/train_3dgs.sh 405841/FRONT PlanA
# bash scripts/train_3dgs.sh 405841 PlanA

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

# ----------------------------------------
# 规范化 scene 名称
# SCENE_REL: 实际相对路径
# SCENE_TAG: 用于输出目录命名
# ----------------------------------------
case "$SCENE_IN" in
  DL3DV-2)
    SCENE_REL="DL3DV-2"
    SCENE_TAG="DL3DV-2"
    ;;
  Re10k-1)
    SCENE_REL="Re10k-1"
    SCENE_TAG="Re10k-1"
    ;;
  405841_FRONT|405841/FRONT|405841)
    SCENE_REL="405841/FRONT"
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

is_valid_scene_dir() {
  local dir="$1"
  [ -d "$dir" ] && [ -d "$dir/images" ] && [ -d "$dir/sparse" ]
}

prepare_sparse_zero_if_needed() {
  local dir="$1"

  if [ -d "$dir/sparse/0" ]; then
    return 0
  fi

  if [ -f "$dir/sparse/cameras.bin" ] || [ -f "$dir/sparse/cameras.txt" ]; then
    mkdir -p "$dir/sparse/0"

    for stem in cameras images points3D; do
      for ext in bin txt; do
        if [ -f "$dir/sparse/${stem}.${ext}" ] && [ ! -e "$dir/sparse/0/${stem}.${ext}" ]; then
          ln -s "../${stem}.${ext}" "$dir/sparse/0/${stem}.${ext}"
        fi
      done
    done

    echo "[INFO] Created sparse/0 symlink layout under: $dir"
    return 0
  fi

  return 1
}

find_source_dir() {
  local candidates=(
    "$ROOT/data/$PLAN/$SCENE_REL"
    "$ROOT/data/$SCENE_REL/$PLAN"
    "$ROOT/data/$SCENE_REL"
    "$ROOT/scenes_3dgs/$PLAN/$SCENE_REL"
    "$ROOT/scenes_3dgs/$SCENE_REL/$PLAN"
    "$ROOT/scenes_3dgs/$SCENE_REL"
  )

  local d
  for d in "${candidates[@]}"; do
    if is_valid_scene_dir "$d"; then
      echo "$d"
      return 0
    fi
  done

  return 1
}

SRC_DIR="$(find_source_dir || true)"

if [ -z "${SRC_DIR:-}" ]; then
  echo "Error: cannot find a valid source scene directory for:"
  echo "  SCENE_IN   = $SCENE_IN"
  echo "  SCENE_REL  = $SCENE_REL"
  echo "  PLAN       = $PLAN"
  echo
  echo "Tried paths like:"
  echo "  $ROOT/data/$SCENE_REL"
  echo "  $ROOT/data/$PLAN/$SCENE_REL"
  echo "  $ROOT/data/$SCENE_REL/$PLAN"
  echo "  $ROOT/scenes_3dgs/$SCENE_REL"
  exit 1
fi

if ! prepare_sparse_zero_if_needed "$SRC_DIR"; then
  echo "Error: COLMAP sparse model not found in expected format."
  echo "Need either:"
  echo "  $SRC_DIR/sparse/0/{cameras,images,points3D}.{bin|txt}"
  echo "or:"
  echo "  $SRC_DIR/sparse/{cameras,images,points3D}.{bin|txt}"
  exit 1
fi

OUT_DIR="$ROOT/outputs/3dgs/$PLAN/$SCENE_TAG"
LOG_DIR="$OUT_DIR/logs"
mkdir -p "$LOG_DIR"

START_TIME="$(date '+%Y-%m-%d %H:%M:%S')"
START_EPOCH="$(date +%s)"

{
  echo "========================================"
  echo "SCENE_IN       : $SCENE_IN"
  echo "SCENE_REL      : $SCENE_REL"
  echo "SCENE_TAG      : $SCENE_TAG"
  echo "PLAN           : $PLAN"
  echo "SRC_DIR        : $SRC_DIR"
  echo "OUT_DIR        : $OUT_DIR"
  echo "REPO           : $REPO"
  echo "START_TIME     : $START_TIME"
  echo "========================================"
} | tee "$LOG_DIR/train_meta.txt"

cd "$REPO"

python train.py \
  -s "$SRC_DIR" \
  -m "$OUT_DIR" \
  --eval \
  --test_iterations 1000 3000 7000 15000 30000 \
  --save_iterations 1000 3000 7000 15000 30000 \
  --checkpoint_iterations 7000 15000 30000 \
  2>&1 | tee "$LOG_DIR/train_console.log"

END_TIME="$(date '+%Y-%m-%d %H:%M:%S')"
END_EPOCH="$(date +%s)"
ELAPSED_SEC=$((END_EPOCH - START_EPOCH))

{
  echo
  echo "END_TIME       : $END_TIME"
  echo "ELAPSED_SEC    : $ELAPSED_SEC"
} | tee -a "$LOG_DIR/train_meta.txt"