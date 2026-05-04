#!/bin/bash

set -euo pipefail

MODEL_DIR="${1:-}"
ITERATION="${2:-}"
if [ -z "$MODEL_DIR" ]; then
  echo "Usage: bash part3/scripts/eval_part3_pretrained.sh /path/to/your/model_dir [ITERATION]"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "$SCRIPT_DIR/eval_part3_3dgs.sh" "$MODEL_DIR" "$ITERATION"
