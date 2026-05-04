#!/bin/bash

set -euo pipefail

SCENE="${1:-}"
PROMPT="${2:-novel view interpolation, geometrically consistent scene, stable camera motion}"
if [ -z "$SCENE" ]; then
  echo "Usage: bash part3/scripts/run_part3_pipeline.sh [SCENE] [PROMPT]"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PART3_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
python3 "$PART3_ROOT/apps/run_part3_pipeline.py" --scene "$SCENE" --prompt "$PROMPT"
