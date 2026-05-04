#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PART3_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$PART3_ROOT/.." && pwd)"

DEFAULT_SCENE="Re10k-1"
SCENE="${SCENE:-$DEFAULT_SCENE}"
SCENES="${SCENES:-}"
VARIANTS="${VARIANTS:-raw conf full}"

NUM_INTERMEDIATE_VIEWS="${NUM_INTERMEDIATE_VIEWS:-6}"
MAX_PAIRS="${MAX_PAIRS:-0}"
KEEP_RATIO="${KEEP_RATIO:-0.8}"
KEEP_RATIO_TAG="${KEEP_RATIO_TAG:-$(printf '%s' "$KEEP_RATIO" | tr '.' 'p')}"
SEED="${SEED:-123}"

SOURCE_RUN_ID="${SOURCE_RUN_ID:-}"
SOURCE_RUN_ID_TEMPLATE="${SOURCE_RUN_ID_TEMPLATE:-}"
RUN_PREFIX="${RUN_PREFIX:-}"
RUN_PREFIX_TEMPLATE="${RUN_PREFIX_TEMPLATE:-}"

if [ -z "$SOURCE_RUN_ID_TEMPLATE" ]; then
  SOURCE_RUN_ID_TEMPLATE="{scene}_shared_pseudo_N{num}_kr{keep}_res{res}"
fi

if [ -z "$RUN_PREFIX_TEMPLATE" ]; then
  RUN_PREFIX_TEMPLATE="{scene}"
fi

SOURCE_CONFIG="${SOURCE_CONFIG:-$PART3_ROOT/configs/project_gen_full.json}"
RAW_CONFIG="${RAW_CONFIG:-$PART3_ROOT/configs/project_gen_raw.json}"
CONF_CONFIG="${CONF_CONFIG:-$PART3_ROOT/configs/project_gen_conf.json}"
FULL_CONFIG="${FULL_CONFIG:-$PART3_ROOT/configs/project_gen_full.json}"
DYNAMICRAFTER_RESOLUTION="${DYNAMICRAFTER_RESOLUTION:-$(python3 -c "import json, sys; data=json.load(open(sys.argv[1])); print(data.get('defaults', {}).get('dynami_crafter', {}).get('resolution', '384_512'))" "$SOURCE_CONFIG")}"
RESOLUTION_TAG="${RESOLUTION_TAG:-$(printf '%s' "$DYNAMICRAFTER_RESOLUTION" | tr '_' 'x')}"

PREP_ENV="${PREP_ENV:-dust3r}"
GEN_ENV="${GEN_ENV:-dynamicrafter}"
BUILD_ENV="${BUILD_ENV:-dust3r}"

REBUILD_VARIANTS="${REBUILD_VARIANTS:-0}"

usage() {
  cat <<EOF
Build Part 3 raw/conf/full ablation scenes by reusing one shared pseudo-view generation per dataset.

Usage:
  bash part3/scripts/run_reuse_pseudo_ablation.sh

Default single-scene run:
  SCENE=$SCENE

Supported scene keys:
  405841_FRONT
  DL3DV-2
  Re10k-1

Run all three datasets:
  SCENES="405841_FRONT DL3DV-2 Re10k-1" bash part3/scripts/run_reuse_pseudo_ablation.sh

Defaults:
  VARIANTS="$VARIANTS"
  NUM_INTERMEDIATE_VIEWS=$NUM_INTERMEDIATE_VIEWS
  MAX_PAIRS=$MAX_PAIRS
  KEEP_RATIO=$KEEP_RATIO
  SEED=$SEED
  DYNAMICRAFTER_RESOLUTION=$DYNAMICRAFTER_RESOLUTION
  SOURCE_RUN_ID_TEMPLATE=$SOURCE_RUN_ID_TEMPLATE
  RUN_PREFIX_TEMPLATE=$RUN_PREFIX_TEMPLATE

Variant outputs for each scene:
  <scene>_gen_raw
  <scene>_gen_conf
  <scene>_gen_full

Useful overrides:
  SCENE=DL3DV-2 bash part3/scripts/run_reuse_pseudo_ablation.sh
  SCENES="405841_FRONT DL3DV-2 Re10k-1" REBUILD_VARIANTS=1 bash part3/scripts/run_reuse_pseudo_ablation.sh
  VARIANTS="conf full" bash part3/scripts/run_reuse_pseudo_ablation.sh

Notes:
  SOURCE_RUN_ID, if set, is used literally for every selected scene.
  Otherwise SOURCE_RUN_ID_TEMPLATE is expanded with {scene}, {num}, {keep}, and {res}.
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

format_template() {
  local template="$1"
  local scene="$2"
  local result="$template"
	  result="${result//\{scene\}/$scene}"
	  result="${result//\{num\}/$NUM_INTERMEDIATE_VIEWS}"
	  result="${result//\{keep\}/$KEEP_RATIO_TAG}"
	  result="${result//\{res\}/$RESOLUTION_TAG}"
	  echo "$result"
}

source_run_id_for_scene() {
  local scene="$1"
  if [ -n "$SOURCE_RUN_ID" ]; then
    echo "$SOURCE_RUN_ID"
  else
    format_template "$SOURCE_RUN_ID_TEMPLATE" "$scene"
  fi
}

run_prefix_for_scene() {
  local scene="$1"
  if [ -n "$RUN_PREFIX" ]; then
    echo "$RUN_PREFIX"
  else
    format_template "$RUN_PREFIX_TEMPLATE" "$scene"
  fi
}

variant_matches_source() {
  local target_pseudo="$1"
  local source_trajectory="$2"
  local source_pseudo="$3"
  if [ ! -f "$target_pseudo" ]; then
    return 1
  fi

  python3 -c \
    "import json, sys; data=json.load(open(sys.argv[1])); derived=data.get('derived_from') or {}; ok=(derived.get('source_trajectory_manifest_path') == sys.argv[2] and derived.get('source_pseudo_manifest_path') == sys.argv[3]); sys.exit(0 if ok else 1)" \
    "$target_pseudo" "$source_trajectory" "$source_pseudo"
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

source_pseudo_resolution_matches() {
  local source_pseudo="$1"
  if [ ! -f "$source_pseudo" ]; then
    return 0
  fi

  python3 -c \
    "import json, sys; data=json.load(open(sys.argv[1])); got=str((data.get('dynami_crafter') or {}).get('resolution', '')); sys.exit(0 if got == sys.argv[2] else 1)" \
    "$source_pseudo" "$DYNAMICRAFTER_RESOLUTION"
}

assert_source_pseudo_resolution() {
  local source_pseudo="$1"
  if source_pseudo_resolution_matches "$source_pseudo"; then
    return
  fi
  echo "ERROR: Existing source pseudo manifest has a different DynamiCrafter resolution." >&2
  echo "       source_pseudo: $source_pseudo" >&2
  echo "       expected: $DYNAMICRAFTER_RESOLUTION" >&2
  echo "       Use a new SOURCE_RUN_ID or delete/regenerate that source run." >&2
  exit 1
}

build_variant() {
  local scene="$1"
  local run_prefix="$2"
  local source_trajectory="$3"
  local source_pseudo="$4"
  local variant="$5"
  local run_id="${run_prefix}_gen_${variant}"
  local config
  config="$(variant_config "$variant")"
  local target_run_dir="$PART3_ROOT/workspace/runs/$scene/$run_id"
  local target_trajectory="$target_run_dir/trajectory_manifest.json"
  local target_pseudo="$target_run_dir/pseudo_views/pseudo_manifest.json"
  local confidence_manifest="$target_run_dir/confidence/confidence_manifest.json"
  local hybrid_scene_dir="$PART3_ROOT/workspace/hybrid_scenes/$run_id"
  local hybrid_manifest="$hybrid_scene_dir/part3_hybrid_manifest.json"

  echo
  echo "==> Building variant: $run_id"
  echo "==> Config: $config"

  local rebuild_this_variant="$REBUILD_VARIANTS"
  if ! variant_matches_source "$target_pseudo" "$source_trajectory" "$source_pseudo"; then
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
      --source-trajectory-manifest "$source_trajectory" \
      --source-pseudo-manifest "$source_pseudo" \
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

build_scene() {
  local scene="$1"
  local source_run_id
  source_run_id="$(source_run_id_for_scene "$scene")"
  local run_prefix
  run_prefix="$(run_prefix_for_scene "$scene")"
  local source_run_dir="$PART3_ROOT/workspace/runs/$scene/$source_run_id"
  local source_trajectory="$source_run_dir/trajectory_manifest.json"
  local source_pseudo="$source_run_dir/pseudo_views/pseudo_manifest.json"

  echo
  echo "================================================================================"
  echo "==> Scene: $scene"
  echo "==> Shared source run: $source_run_id"
  echo "==> Shared trajectory: $source_trajectory"
  echo "==> Shared pseudo manifest: $source_pseudo"
  echo "================================================================================"

  run_if_missing "$source_trajectory" "$PREP_ENV" \
    python3 "$PART3_ROOT/apps/prepare_scene.py" \
      --config "$SOURCE_CONFIG" \
      --scene "$scene" \
      --run-id "$source_run_id" \
      --num-intermediate-views "$NUM_INTERMEDIATE_VIEWS" \
      --max-pairs "$MAX_PAIRS"

  run_if_missing "$source_pseudo" "$GEN_ENV" \
    python3 "$PART3_ROOT/apps/generate_pseudo_views.py" \
      --config "$SOURCE_CONFIG" \
      --trajectory-manifest "$source_trajectory" \
      --keep-ratio "$KEEP_RATIO" \
      --seed "$SEED" \
      --resolution "$DYNAMICRAFTER_RESOLUTION"
  assert_source_pseudo_resolution "$source_pseudo"

  local variant
  for variant in $VARIANTS; do
    build_variant "$scene" "$run_prefix" "$source_trajectory" "$source_pseudo" "$variant"
  done
}

init_conda

echo "==> Project root: $PROJECT_ROOT"
echo "==> Selected variants: $VARIANTS"
echo "==> DynamiCrafter resolution: $DYNAMICRAFTER_RESOLUTION"

cd "$PROJECT_ROOT"

if [ -n "$SCENES" ]; then
  SCENES="${SCENES//,/ }"
  read -r -a SELECTED_SCENES <<< "$SCENES"
else
  SELECTED_SCENES=("$SCENE")
fi

for selected_scene in "${SELECTED_SCENES[@]}"; do
  build_scene "$selected_scene"
done

echo
echo "==> Reusable pseudo-view ablation build complete."
echo "==> Training can now use the generated hybrid scenes with absolute confidence_manifest paths."
