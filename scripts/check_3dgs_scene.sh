#!/bin/bash
# 用法:
#   bash scripts/check_3dgs_scene.sh DL3DV-2
#   bash scripts/check_3dgs_scene.sh Re10k-1
#   bash scripts/check_3dgs_scene.sh 405841_FRONT

set -euo pipefail

# =========================================================
# 0. Path Config (集中管理，后续只改这里)
# =========================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# 你的数据根目录
DATA_ROOT="$PROJECT_ROOT/data"

# 3DGS 输出目录
GS_OUTPUT_ROOT="$PROJECT_ROOT/outputs/3dgs"

# 临时导出 COLMAP txt 的目录
TMP_ROOT="/tmp/check_3dgs_scene"

# 是否打印目录树
SHOW_TREE="1"

# tree 显示层数
TREE_DEPTH="2"

# 默认场景名
DEFAULT_SCENE="DL3DV-2"

# =========================================================
# 1. Scene Selection
# =========================================================

SCENE="${1:-$DEFAULT_SCENE}"

# 场景名 -> 实际目录
# 后续如果你的目录结构改了，只需要改这里
case "$SCENE" in
  DL3DV-2)
    SCENE_DIR="$DATA_ROOT/DL3DV-2"
    ;;
  Re10k-1)
    SCENE_DIR="$DATA_ROOT/Re10k-1"
    ;;
  405841_FRONT)
    SCENE_DIR="$DATA_ROOT/405841/FRONT"
    ;;
  *)
    echo "Error: unknown scene name: $SCENE"
    echo "Supported scenes:"
    echo "  - DL3DV-2"
    echo "  - Re10k-1"
    echo "  - 405841_FRONT"
    exit 1
    ;;
esac

# =========================================================
# 2. Derived Paths
# =========================================================

IMAGES_DIR="$SCENE_DIR/images"
RGB_DIR="$SCENE_DIR/rgb"
INPUT_DIR="$SCENE_DIR/input"
DISTORTED_DIR="$SCENE_DIR/distorted"

CAMERAS_JSON="$SCENE_DIR/cameras.json"
INTRINSICS_JSON="$SCENE_DIR/intrinsics.json"

SPARSE_ROOT="$SCENE_DIR/sparse"
SPARSE_MODEL_DIR=""
TXT_DIR="$TMP_ROOT/$SCENE"

# =========================================================
# 3. Helpers
# =========================================================

print_header() {
  echo "========================================"
  echo "$1"
  echo "========================================"
}

check_exists() {
  local path="$1"
  local label="$2"

  if [ -e "$path" ]; then
    echo "[OK]   $label: $path"
  else
    echo "[MISS] $label: $path"
  fi
}

check_dir() {
  local path="$1"
  local label="$2"

  if [ -d "$path" ]; then
    echo "[OK]   $label: $path"
  else
    echo "[MISS] $label: $path"
  fi
}

detect_sparse_model_dir() {
  # 优先用 sparse/0
  if [ -d "$SPARSE_ROOT/0" ]; then
    SPARSE_MODEL_DIR="$SPARSE_ROOT/0"
    return
  fi

  # 如果 sparse/ 下面直接就是 cameras.bin/images.bin/points3D.bin
  if [ -f "$SPARSE_ROOT/cameras.bin" ] || [ -f "$SPARSE_ROOT/cameras.txt" ]; then
    SPARSE_MODEL_DIR="$SPARSE_ROOT"
    return
  fi

  # 尝试寻找 sparse 下的第一个子模型目录
  local first_model=""
  first_model=$(find "$SPARSE_ROOT" -mindepth 1 -maxdepth 1 -type d | sort | head -n 1 || true)
  if [ -n "$first_model" ]; then
    SPARSE_MODEL_DIR="$first_model"
    return
  fi

  SPARSE_MODEL_DIR=""
}

count_files_in_dir() {
  local path="$1"
  if [ -d "$path" ]; then
    find "$path" -maxdepth 1 -type f | wc -l
  else
    echo "0"
  fi
}

# =========================================================
# 4. Basic Info
# =========================================================

print_header "Checking scene"

echo "Scene name        : $SCENE"
echo "Project root      : $PROJECT_ROOT"
echo "Data root         : $DATA_ROOT"
echo "Scene dir         : $SCENE_DIR"
echo "3DGS output root  : $GS_OUTPUT_ROOT"
echo "Temp txt dir      : $TXT_DIR"

if [ ! -d "$SCENE_DIR" ]; then
  echo
  echo "Error: scene directory not found."
  echo "Please check SCENE mapping or DATA_ROOT."
  exit 1
fi

# =========================================================
# 5. Basic Structure Check
# =========================================================

echo
print_header "Basic structure check"

check_dir "$IMAGES_DIR" "images dir"
check_dir "$RGB_DIR" "rgb dir"
check_dir "$SPARSE_ROOT" "sparse root"
check_exists "$CAMERAS_JSON" "cameras.json"
check_exists "$INTRINSICS_JSON" "intrinsics.json"
check_exists "$INPUT_DIR" "input"
check_exists "$DISTORTED_DIR" "distorted"

echo
echo "File statistics:"
echo "  images files    : $(count_files_in_dir "$IMAGES_DIR")"
echo "  rgb files       : $(count_files_in_dir "$RGB_DIR")"

# =========================================================
# 6. Tree Preview
# =========================================================

if [ "$SHOW_TREE" = "1" ]; then
  echo
  print_header "Directory tree"

  if command -v tree >/dev/null 2>&1; then
    tree -L "$TREE_DEPTH" "$SCENE_DIR"
  else
    find "$SCENE_DIR" -maxdepth "$TREE_DEPTH" | sort
  fi
fi

# =========================================================
# 7. Detect COLMAP Sparse Model
# =========================================================

echo
print_header "Detect sparse model"

if [ ! -d "$SPARSE_ROOT" ]; then
  echo "Error: sparse root not found: $SPARSE_ROOT"
  exit 1
fi

detect_sparse_model_dir

if [ -z "$SPARSE_MODEL_DIR" ]; then
  echo "Error: cannot find a valid COLMAP sparse model under:"
  echo "  $SPARSE_ROOT"
  echo
  echo "Expected one of:"
  echo "  - $SPARSE_ROOT/0"
  echo "  - $SPARSE_ROOT containing cameras.bin/images.bin/points3D.bin"
  echo "  - any subfolder under sparse/ that is a valid COLMAP model"
  exit 1
fi

echo "[OK] Using sparse model dir: $SPARSE_MODEL_DIR"

# =========================================================
# 8. Convert COLMAP Model to TXT
# =========================================================

echo
print_header "Convert COLMAP model to TXT"

rm -rf "$TXT_DIR"
mkdir -p "$TXT_DIR"

colmap model_converter \
  --input_path "$SPARSE_MODEL_DIR" \
  --output_path "$TXT_DIR" \
  --output_type TXT

echo "[OK] Converted to: $TXT_DIR"

# =========================================================
# 9. Inspect cameras.txt / images.txt / points3D.txt
# =========================================================

echo
print_header "cameras.txt preview"
head -n 20 "$TXT_DIR/cameras.txt"

CAM_MODEL=$(awk '$1 !~ /^#/ && NF>0 {print $2; exit}' "$TXT_DIR/cameras.txt")
IMG_W=$(awk '$1 !~ /^#/ && NF>0 {print $3; exit}' "$TXT_DIR/cameras.txt")
IMG_H=$(awk '$1 !~ /^#/ && NF>0 {print $4; exit}' "$TXT_DIR/cameras.txt")

REG_IMAGES=$(awk '
  BEGIN{cnt=0}
  $1 ~ /^#/ {next}
  NF==0 {next}
  {cnt++}
  END{print int(cnt/2)}
' "$TXT_DIR/images.txt")

PTS3D=$(awk '
  BEGIN{cnt=0}
  $1 !~ /^#/ && NF>0 {cnt++}
  END{print cnt}
' "$TXT_DIR/points3D.txt")

# =========================================================
# 10. Summary
# =========================================================

echo
print_header "Parsed summary"

echo "Detected camera model : ${CAM_MODEL:-UNKNOWN}"
echo "Image size            : ${IMG_W:-?} x ${IMG_H:-?}"
echo "Registered images     : ${REG_IMAGES:-0}"
echo "Sparse points3D       : ${PTS3D:-0}"

# =========================================================
# 11. Verdict
# =========================================================

echo
print_header "Verdict"

if [ -d "$IMAGES_DIR" ]; then
  TRAIN_SOURCE_DIR="$SCENE_DIR"
elif [ -d "$RGB_DIR" ]; then
  TRAIN_SOURCE_DIR="$SCENE_DIR"
else
  TRAIN_SOURCE_DIR="$SCENE_DIR"
fi

if [ "$CAM_MODEL" = "PINHOLE" ] || [ "$CAM_MODEL" = "SIMPLE_PINHOLE" ]; then
  echo "[PASS] Camera model is compatible with standard 3DGS."
  echo
  echo "Suggested training command:"
  echo "python train.py -s $TRAIN_SOURCE_DIR -m $GS_OUTPUT_ROOT/$SCENE --eval"
else
  echo "[WARN] Camera model is: ${CAM_MODEL:-UNKNOWN}"
  echo "This usually means standard 3DGS may need undistortion / convert.py first."
  echo
  echo "You may need to:"
  echo "  1) run convert.py"
  echo "  2) or undistort COLMAP cameras first"
fi