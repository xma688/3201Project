#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PART3_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$PART3_ROOT/.." && pwd)"

SCENE="${SCENE:-Re10k-1}"
DEFAULT_RUN_ID="${RUN_ID:-Re10k-1_gen_full}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  SHOW_HELP=1
else
  SHOW_HELP=0
fi

RUN_ID="${1:-$DEFAULT_RUN_ID}"
ITERATIONS="${2:-${ITERATIONS:-30000}}"

CONFIG="${CONFIG:-$PART3_ROOT/configs/project_gen_full.json}"
NUM_INTERMEDIATE_VIEWS="${NUM_INTERMEDIATE_VIEWS:-6}"
MAX_PAIRS="${MAX_PAIRS:-0}"
KEEP_RATIO="${KEEP_RATIO:-0.8}"
SEED="${SEED:-123}"

PREP_ENV="${PREP_ENV:-dust3r}"
GEN_ENV="${GEN_ENV:-dynamicrafter}"
CONF_ENV="${CONF_ENV:-dust3r}"
HYBRID_ENV="${HYBRID_ENV:-dust3r}"
TRAIN_ENV="${TRAIN_ENV:-3dgs}"

HYBRID_NAME="${HYBRID_NAME:-$RUN_ID}"
OUTPUT_TAG="${OUTPUT_TAG:-$RUN_ID}"

TRAJECTORY_MANIFEST="$PART3_ROOT/workspace/runs/$SCENE/$RUN_ID/trajectory_manifest.json"
PSEUDO_MANIFEST="$PART3_ROOT/workspace/runs/$SCENE/$RUN_ID/pseudo_views/pseudo_manifest.json"
CONFIDENCE_MANIFEST="$PART3_ROOT/workspace/runs/$SCENE/$RUN_ID/confidence/confidence_manifest.json"
HYBRID_SCENE_DIR="$PART3_ROOT/workspace/hybrid_scenes/$HYBRID_NAME"
HYBRID_MANIFEST="$HYBRID_SCENE_DIR/part3_hybrid_manifest.json"
OUTPUT_DIR="$PART3_ROOT/workspace/3dgs_outputs/$OUTPUT_TAG"

REUSE_EXISTING="${REUSE_EXISTING:-1}"

usage() {
  cat <<EOF
Run Re10k Sparse + Generated + Confidence + Consistency.

Usage:
  bash part3/scripts/run_re10k_gen_full.sh [RUN_ID] [ITERATIONS]

Defaults:
  RUN_ID=$RUN_ID
  ITERATIONS=$ITERATIONS
  SCENE=$SCENE
  CONFIG=$CONFIG
  NUM_INTERMEDIATE_VIEWS=$NUM_INTERMEDIATE_VIEWS
  KEEP_RATIO=$KEEP_RATIO
  MAX_PAIRS=$MAX_PAIRS

Useful overrides:
  SCENE=Re10k-1 RUN_ID=Re10k-1_gen_full_v2 ITERATIONS=3000 bash ...
  REUSE_EXISTING=0 bash ...   # rebuild manifests instead of skipping existing ones
EOF
}

if [ "$SHOW_HELP" = "1" ]; then
  RUN_ID="$DEFAULT_RUN_ID"
  usage
  exit 0
fi

init_conda() {
  if command -v conda >/dev/null 2>&1; then
    return
  fi

  local candidates=(
    "$HOME/miniconda3/etc/profile.d/conda.sh"
    "$HOME/anaconda3/etc/profile.d/conda.sh"
    "/opt/miniconda3/etc/profile.d/conda.sh"
    "/opt/conda/etc/profile.d/conda.sh"
  )
  local conda_sh
  for conda_sh in "${candidates[@]}"; do
    if [ -f "$conda_sh" ]; then
      # shellcheck source=/dev/null
      source "$conda_sh"
      return
    fi
  done

  echo "Could not find conda. Please activate conda first or set PATH accordingly." >&2
  exit 1
}

run_in_env() {
  local env_name="$1"
  shift

  echo
  echo "==> [$env_name] $*"
  conda run --no-capture-output -n "$env_name" "$@"
}

run_or_skip() {
  local target="$1"
  local env_name="$2"
  shift 2

  if [ "$REUSE_EXISTING" = "1" ] && [ -e "$target" ]; then
    echo
    echo "==> Reusing existing: $target"
    return
  fi

  run_in_env "$env_name" "$@"
}

init_conda

echo "==> Project root: $PROJECT_ROOT"
echo "==> Run id: $RUN_ID"
echo "==> Output tag: $OUTPUT_TAG"

cd "$PROJECT_ROOT"

run_or_skip "$TRAJECTORY_MANIFEST" "$PREP_ENV" \
  python3 "$PART3_ROOT/apps/prepare_scene.py" \
    --config "$CONFIG" \
    --scene "$SCENE" \
    --run-id "$RUN_ID" \
    --num-intermediate-views "$NUM_INTERMEDIATE_VIEWS" \
    --max-pairs "$MAX_PAIRS"

run_or_skip "$PSEUDO_MANIFEST" "$GEN_ENV" \
  python3 "$PART3_ROOT/apps/generate_pseudo_views.py" \
    --config "$CONFIG" \
    --trajectory-manifest "$TRAJECTORY_MANIFEST" \
    --keep-ratio "$KEEP_RATIO" \
    --seed "$SEED"

run_or_skip "$CONFIDENCE_MANIFEST" "$CONF_ENV" \
  python3 "$PART3_ROOT/apps/build_confidence.py" \
    --config "$CONFIG" \
    --pseudo-manifest "$PSEUDO_MANIFEST" \
    --enable-clip-consistency \
    --enable-patch-pruning \
    --patch-size 16 \
    --patch-threshold 0.25

run_or_skip "$HYBRID_MANIFEST" "$HYBRID_ENV" \
  python3 "$PART3_ROOT/apps/build_hybrid_scene.py" \
    --config "$CONFIG" \
    --trajectory-manifest "$TRAJECTORY_MANIFEST" \
    --pseudo-manifest "$PSEUDO_MANIFEST" \
    --confidence-manifest "$CONFIDENCE_MANIFEST" \
    --hybrid-name "$HYBRID_NAME"

run_in_env "$TRAIN_ENV" \
  bash "$PART3_ROOT/scripts/train_part3_3dgs.sh" \
    "$HYBRID_SCENE_DIR" \
    "$OUTPUT_TAG" \
    "$ITERATIONS" \
    --confidence_manifest "$CONFIDENCE_MANIFEST" \
    --enable_online_confidence \
    --confidence_refresh_interval 500 \
    --confidence_writeback_interval 1000 \
    --online_rgb_sigma 0.2 \
    --online_feature_sigma 0.25 \
    --online_patch_size 16 \
    --online_patch_threshold 0.25 \
    --online_patch_low_weight 0.15 \
    --online_patch_min_keep_ratio 0.1 \
    --diagnostics_interval 1000 \
    --diagnostics_debug_views 4 \
    --pseudo_warmup_iters 2000 \
    --pseudo_full_iters 8000 \
    --pseudo_ratio_mid 0.25 \
    --pseudo_ratio_final 0.5 \
    --pseudo_weight_mid 0.2 \
    --pseudo_weight_final 0.5 \
    --pseudo_lpips_weight 0.05

echo
echo "==> Done."
echo "==> Hybrid scene: $HYBRID_SCENE_DIR"
echo "==> Training output: $OUTPUT_DIR"
