#!/bin/bash
# bash scripts/organize_3dgs_scene.sh Re10k-1 0
# bash scripts/organize_3dgs_scene.sh DL3DV-2 0
# bash scripts/organize_3dgs_scene.sh 405841_FRONT 0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SCENE="${1:-DL3DV-2}"      # 405841_FRONT | DL3DV-2 | Re10k-1
MODEL_IDX="${2:-1}"        # COLMAP sparse model id, usually 0 or 1

case "$SCENE" in
  405841_FRONT)
    IMG_DIR="$ROOT/data/405841/FRONT/rgb"
    COLMAP_DIR="$ROOT/outputs/colmap/405841_FRONT"
    ;;
  DL3DV-2)
    IMG_DIR="$ROOT/data/DL3DV-2/rgb"
    COLMAP_DIR="$ROOT/outputs/colmap/DL3DV-2"
    ;;
  Re10k-1)
    IMG_DIR="$ROOT/data/Re10k-1/images"
    COLMAP_DIR="$ROOT/outputs/colmap/Re10k-1"
    ;;
  *)
    echo "Unsupported scene: $SCENE"
    echo "Supported scenes: 405841_FRONT, DL3DV-2, Re10k-1"
    exit 1
    ;;
esac

OUT="$ROOT/scenes_3dgs/$SCENE"
SPARSE_SRC="$COLMAP_DIR/sparse/$MODEL_IDX"
SPARSE_DST="$OUT/sparse/0"
TXT_DIR="/tmp/${SCENE}_txt"

echo "========================================"
echo "SCENE      : $SCENE"
echo "IMG_DIR    : $IMG_DIR"
echo "COLMAP_DIR : $COLMAP_DIR"
echo "MODEL_IDX  : $MODEL_IDX"
echo "OUT        : $OUT"
echo "========================================"

if [ ! -d "$IMG_DIR" ]; then
  echo "Error: image directory not found: $IMG_DIR"
  exit 1
fi

if [ ! -d "$SPARSE_SRC" ]; then
  echo "Error: sparse model not found: $SPARSE_SRC"
  echo "Available sparse models:"
  ls -lah "$COLMAP_DIR/sparse" || true
  exit 1
fi

mkdir -p "$OUT/sparse"

if [ -e "$OUT/images" ]; then
  rm -rf "$OUT/images"
fi
ln -s "$IMG_DIR" "$OUT/images"

if [ -d "$SPARSE_DST" ]; then
  rm -rf "$SPARSE_DST"
fi
cp -r "$SPARSE_SRC" "$SPARSE_DST"

rm -rf "$TXT_DIR"
mkdir -p "$TXT_DIR"

colmap model_converter \
  --input_path "$SPARSE_DST" \
  --output_path "$TXT_DIR" \
  --output_type TXT

echo
echo "===== cameras.txt ====="
head -n 20 "$TXT_DIR/cameras.txt"

CAM_MODEL=$(awk '$1 !~ /^#/ && NF>0 {print $2; exit}' "$TXT_DIR/cameras.txt")
echo
echo "Detected camera model: $CAM_MODEL"

if [ "$CAM_MODEL" != "PINHOLE" ] && [ "$CAM_MODEL" != "SIMPLE_PINHOLE" ]; then
  echo "WARNING: camera model is $CAM_MODEL"
  echo "You may need convert.py / undistortion before training 3DGS."
fi

echo
echo "===== final structure ====="
find "$OUT" -maxdepth 3 | sort

echo
echo "Done. 3DGS source folder is ready:"
echo "$OUT"