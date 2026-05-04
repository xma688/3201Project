#!/bin/bash

set -euo pipefail

BASELINE_DIR="${1:-}"
PART3_DIR="${2:-}"

if [ -z "$BASELINE_DIR" ] || [ -z "$PART3_DIR" ]; then
  echo "Usage: bash part3/scripts/compare_sparse_vs_generated.sh /path/to/your/baseline_model /path/to/your/part3_model"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PART3_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

python3 "$PART3_ROOT/apps/compare_part3_metrics.py" \
  --baseline "$BASELINE_DIR" \
  --part3 "$PART3_DIR" \
  --output "$PART3_DIR/part3_compare_summary.json"
