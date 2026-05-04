#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PART3_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$PART3_ROOT/.." && pwd)"

if [ "${PART3_USE_LEGACY_RE10K_REUSE_SCRIPT:-0}" != "1" ]; then
  if [ -z "${SCENE:-}" ] && [ -z "${SCENES:-}" ]; then
    export SCENE="Re10k-1"
  fi
  exec bash "$SCRIPT_DIR/run_reuse_pseudo_ablation.sh" "$@"
fi

SCENE="${SCENE:-Re10k-1}"
RUN_PREFIX="${RUN_PREFIX:-$SCENE}"
NUM_INTERMEDIATE_VIEWS="${NUM_INTERMEDIATE_VIEWS:-6}"
MAX_PAIRS="${MAX_PAIRS:-0}"
KEEP_RATIO="${KEEP_RATIO:-0.8}"
SEED="${SEED:-123}"

SOURCE_RUN_ID="${SOURCE_RUN_ID:-${SCENE}_shared_pseudo_N${NUM_INTERMEDIATE_VIEWS}_kr0p8}"
SOURCE_CONFIG="${SOURCE_CONFIG:-$PART3_ROOT/configs/project_gen_full.json}"
RAW_CONFIG="${RAW_CONFIG:-$PART3_ROOT/configs/project_gen_raw.json}"
CONF_CONFIG="${CONF_CONFIG:-$PART3_ROOT/configs/project_gen_conf.json}"
FULL_CONFIG="${FULL_CONFIG:-$PART3_ROOT/configs/project_gen_full.json}"

PREP_ENV="${PREP_ENV:-dust3r}"
GEN_ENV="${GEN_ENV:-dynamicrafter}"
BUILD_ENV="${BUILD_ENV:-dust3r}"

REBUILD_VARIANTS="${REBUILD_VARIANTS:-0}"

SOURCE_RUN_DIR="$PART3_ROOT/workspace/runs/$SCENE/$SOURCE_RUN_ID"
SOURCE_TRAJECTORY="$SOURCE_RUN_DIR/trajectory_manifest.json"
SOURCE_PSEUDO="$SOURCE_RUN_DIR/pseudo_views/pseudo_manifest.json"

usage() {
  cat <<EOF
Build Re10k raw/conf/full ablation scenes by reusing one shared pseudo-view generation.

Usage:
  bash part3/scripts/run_re10k_reuse_pseudo_ablation.sh

Defaults:
  SCENE=$SCENE
  SOURCE_RUN_ID=$SOURCE_RUN_ID
  NUM_INTERMEDIATE_VIEWS=$NUM_INTERMEDIATE_VIEWS
  MAX_PAIRS=$MAX_PAIRS
  KEEP_RATIO=$KEEP_RATIO
  SEED=$SEED

Outputs:
  $PART3_ROOT/workspace/runs/$SCENE/${RUN_PREFIX}_gen_raw
  $PART3_ROOT/workspace/runs/$SCENE/${RUN_PREFIX}_gen_conf
  $PART3_ROOT/workspace/runs/$SCENE/${RUN_PREFIX}_gen_full
  $PART3_ROOT/workspace/hybrid_scenes/${RUN_PREFIX}_gen_raw
  $PART3_ROOT/workspace/hybrid_scenes/${RUN_PREFIX}_gen_conf
  $PART3_ROOT/workspace/hybrid_scenes/${RUN_PREFIX}_gen_full

Useful overrides:
  SOURCE_RUN_ID=Re10k-1_shared_pseudo_N6_kr0p8 bash part3/scripts/run_re10k_reuse_pseudo_ablation.sh
  REBUILD_VARIANTS=1 bash part3/scripts/run_re10k_reuse_pseudo_ablation.sh

Notes:
  REBUILD_VARIANTS=1 only removes per-variant confidence/ and hybrid_scenes/<variant>.
  It never deletes the shared source pseudo frames.
  Training is intentionally not launched by this script.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
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

run_if_missing() {
  local target="$1"
  local env_name="$2"
  shift 2

  if [ -e "$target" ]; then
    echo
    echo "==> Reusing existing: $target"
    return
  fi

  run_in_env "$env_name" "$@"
}

variant_matches_source() {
  local target_pseudo="$1"
  if [ ! -f "$target_pseudo" ]; then
    return 1
  fi

  python3 -c \
    "import json, sys; data=json.load(open(sys.argv[1])); derived=data.get('derived_from') or {}; ok=(derived.get('source_trajectory_manifest_path') == sys.argv[2] and derived.get('source_pseudo_manifest_path') == sys.argv[3]); sys.exit(0 if ok else 1)" \
    "$target_pseudo" "$SOURCE_TRAJECTORY" "$SOURCE_PSEUDO"
}

variant_config() {
  case "$1" in
    raw) echo "$RAW_CONFIG" ;;
    conf) echo "$CONF_CONFIG" ;;
    full) echo "$FULL_CONFIG" ;;
    *)
      echo "Unknown variant: $1" >&2
      exit 1
      ;;
  esac
}

confidence_formula_is_current() {
  local confidence_manifest="$1"
  if [ ! -f "$confidence_manifest" ]; then
    return 0
  fi

  python3 -c \
    "import json, sys; data=json.load(open(sys.argv[1])); formula=data.get('confidence_formula') or {}; patch=formula.get('patch') or {}; padding=formula.get('padding') or {}; ok=(formula.get('type') == 'hard_validity_times_soft_floor' and int(formula.get('version', 0)) >= 3 and patch.get('type') == 'soft_weight' and padding.get('type') == 'edge_connected_near_constant_band' and abs(float(formula.get('floor', -1)) - 0.3) < 1e-9); sys.exit(0 if ok else 1)" \
    "$confidence_manifest"
}

build_variant() {
  local variant="$1"
  local run_id="${RUN_PREFIX}_gen_${variant}"
  local config
  config="$(variant_config "$variant")"
  local target_run_dir="$PART3_ROOT/workspace/runs/$SCENE/$run_id"
  local target_trajectory="$target_run_dir/trajectory_manifest.json"
  local target_pseudo="$target_run_dir/pseudo_views/pseudo_manifest.json"
  local confidence_manifest="$target_run_dir/confidence/confidence_manifest.json"
  local hybrid_scene_dir="$PART3_ROOT/workspace/hybrid_scenes/$run_id"
  local hybrid_manifest="$hybrid_scene_dir/part3_hybrid_manifest.json"

  echo
  echo "==> Building variant: $run_id"
  echo "==> Config: $config"

  local rebuild_this_variant="$REBUILD_VARIANTS"
  if ! variant_matches_source "$target_pseudo"; then
    echo "==> Existing variant is missing or derived from a different source; rebuilding confidence/hybrid"
    rebuild_this_variant=1
  fi
  if ! confidence_formula_is_current "$confidence_manifest"; then
    echo "==> Existing confidence manifest uses an old formula; rebuilding confidence/hybrid"
    rebuild_this_variant=1
  fi

  if [ "$rebuild_this_variant" = "1" ]; then
    echo "==> Removing per-variant confidence and hybrid scene only"
    rm -rf "$target_run_dir/confidence" "$hybrid_scene_dir"
  fi

  run_in_env "$BUILD_ENV" \
    python3 "$PART3_ROOT/apps/derive_pseudo_variant.py" \
      --config "$config" \
      --source-trajectory-manifest "$SOURCE_TRAJECTORY" \
      --source-pseudo-manifest "$SOURCE_PSEUDO" \
      --target-run-id "$run_id" \
      --target-run-dir "$target_run_dir" \
      --overwrite

  run_if_missing "$confidence_manifest" "$BUILD_ENV" \
    python3 "$PART3_ROOT/apps/build_confidence.py" \
      --config "$config" \
      --pseudo-manifest "$target_pseudo"

  run_if_missing "$hybrid_manifest" "$BUILD_ENV" \
    python3 "$PART3_ROOT/apps/build_hybrid_scene.py" \
      --config "$config" \
      --trajectory-manifest "$target_trajectory" \
      --pseudo-manifest "$target_pseudo" \
      --confidence-manifest "$confidence_manifest" \
      --hybrid-name "$run_id"

  echo
  echo "==> Variant ready: $run_id"
  echo "    pseudo manifest: $target_pseudo"
  echo "    confidence manifest: $confidence_manifest"
  echo "    hybrid scene: $hybrid_scene_dir"
}

init_conda

echo "==> Project root: $PROJECT_ROOT"
echo "==> Shared source run: $SOURCE_RUN_ID"
echo "==> Shared trajectory: $SOURCE_TRAJECTORY"
echo "==> Shared pseudo manifest: $SOURCE_PSEUDO"

cd "$PROJECT_ROOT"

run_if_missing "$SOURCE_TRAJECTORY" "$PREP_ENV" \
  python3 "$PART3_ROOT/apps/prepare_scene.py" \
    --config "$SOURCE_CONFIG" \
    --scene "$SCENE" \
    --run-id "$SOURCE_RUN_ID" \
    --num-intermediate-views "$NUM_INTERMEDIATE_VIEWS" \
    --max-pairs "$MAX_PAIRS"

run_if_missing "$SOURCE_PSEUDO" "$GEN_ENV" \
  python3 "$PART3_ROOT/apps/generate_pseudo_views.py" \
    --config "$SOURCE_CONFIG" \
    --trajectory-manifest "$SOURCE_TRAJECTORY" \
    --keep-ratio "$KEEP_RATIO" \
    --seed "$SEED"

build_variant raw
build_variant conf
build_variant full

echo
echo "==> Reusable pseudo-view ablation build complete."
echo "==> Training can now use the three hybrid scenes above with absolute confidence_manifest paths."
