#!/bin/bash

set -euo pipefail

SCENE="${1:-}"
PSEUDO_MANIFEST="${2:-}"
if [ -z "$SCENE" ]; then
  echo "Usage: bash part3/scripts/run_part3_pretrained.sh SCENE [PSEUDO_MANIFEST] [HYBRID_NAME] [OUTPUT_TAG]"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PART3_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG="${CONFIG:-$PART3_ROOT/configs/project_pretrained_full.json}"

ARGS=(python3 "$PART3_ROOT/apps/run_part3_pretrained.py" --config "$CONFIG" --scene "$SCENE")
if [ -n "$PSEUDO_MANIFEST" ]; then
  ARGS+=(--pseudo-manifest "$PSEUDO_MANIFEST")
fi
if [ -n "${3:-}" ]; then
  ARGS+=(--hybrid-name "$3")
fi
if [ -n "${4:-}" ]; then
  ARGS+=(--output-tag "$4")
fi

"${ARGS[@]}"
