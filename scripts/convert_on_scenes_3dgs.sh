#!/bin/bash
# 用法:
# bash scripts/convert_on_scenes_3dgs.sh Re10k-1
# bash scripts/convert_on_scenes_3dgs.sh DL3DV-2
# bash scripts/convert_on_scenes_3dgs.sh 405841_FRONT

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SCENE="${1:-DL3DV-2}"
SCENE_DIR="$ROOT/scenes_3dgs/$SCENE"

case "$SCENE" in
  405841_FRONT)
    COLMAP_DIR="$ROOT/outputs/colmap/405841_FRONT"
    ;;
  DL3DV-2)
    COLMAP_DIR="$ROOT/outputs/colmap/DL3DV-2"
    ;;
  Re10k-1)
    COLMAP_DIR="$ROOT/outputs/colmap/Re10k-1"
    ;;
  *)
    echo "Unsupported scene: $SCENE"
    exit 1
    ;;
esac

echo "========================================"
echo "SCENE      : $SCENE"
echo "SCENE_DIR  : $SCENE_DIR"
echo "COLMAP_DIR : $COLMAP_DIR"
echo "========================================"

if [ ! -d "$SCENE_DIR" ]; then
  echo "Error: scene dir not found: $SCENE_DIR"
  exit 1
fi

if [ ! -e "$SCENE_DIR/images" ]; then
  echo "Error: images not found: $SCENE_DIR/images"
  exit 1
fi

if [ ! -d "$SCENE_DIR/sparse/0" ]; then
  echo "Error: sparse model not found: $SCENE_DIR/sparse/0"
  exit 1
fi

if [ ! -f "$COLMAP_DIR/database.db" ]; then
  echo "Error: database.db not found: $COLMAP_DIR/database.db"
  exit 1
fi

# 清理旧的 convert 中间目录，避免重复运行时冲突
rm -rf "$SCENE_DIR/input" "$SCENE_DIR/distorted"

# 1) images -> input
mv "$SCENE_DIR/images" "$SCENE_DIR/input"

# 2) sparse -> distorted/sparse
mkdir -p "$SCENE_DIR/distorted"
mv "$SCENE_DIR/sparse" "$SCENE_DIR/distorted/sparse"

# 3) 拷贝 database.db
cp "$COLMAP_DIR/database.db" "$SCENE_DIR/distorted/database.db"

# 4) 运行 convert.py
cd "$ROOT/gaussian-splatting"
python convert.py -s "$SCENE_DIR" --skip_matching

echo
echo "===== after convert ====="
find "$SCENE_DIR" -maxdepth 3 | sort