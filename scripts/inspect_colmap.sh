#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/inspect_colmap.sh re10k
#   bash scripts/inspect_colmap.sh dl3dv
#   bash scripts/inspect_colmap.sh waymo_front
#
# Optional:
#   bash scripts/inspect_colmap.sh re10k 0     # do not save best txt preview
#   bash scripts/inspect_colmap.sh re10k 1     # save best txt preview (default)

DATASET_KEY="${1:-}"
SAVE_BEST_TXT="${2:-1}"

if [[ -z "${DATASET_KEY}" ]]; then
    echo "Usage: bash scripts/inspect_colmap.sh [re10k|dl3dv|waymo_front] [save_best_txt:0|1]"
    exit 1
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_ROOT="${PROJECT_ROOT}/outputs/colmap"

case "${DATASET_KEY}" in
    re10k)
        SCENE_NAME="Re10k-1"
        ;;
    dl3dv)
        SCENE_NAME="DL3DV-2"
        ;;
    waymo_front)
        SCENE_NAME="405841_FRONT"
        ;;
    *)
        echo "Unknown dataset key: ${DATASET_KEY}"
        echo "Allowed: re10k | dl3dv | waymo_front"
        exit 1
        ;;
esac

OUT_DIR="${OUTPUT_ROOT}/${SCENE_NAME}"
SPARSE_ROOT="${OUT_DIR}/sparse"
BEST_TXT_DIR="${OUT_DIR}/best_sparse_txt"
TMP_ROOT="${OUT_DIR}/_inspect_tmp"

if [[ ! -d "${SPARSE_ROOT}" ]]; then
    echo "Sparse model directory not found: ${SPARSE_ROOT}"
    echo "Run COLMAP first."
    exit 1
fi

rm -rf "${TMP_ROOT}"
mkdir -p "${TMP_ROOT}"

echo "========================================"
echo "SCENE_NAME   : ${SCENE_NAME}"
echo "OUT_DIR      : ${OUT_DIR}"
echo "SPARSE_ROOT  : ${SPARSE_ROOT}"
echo "========================================"
echo

BEST_DIR=""
BEST_REG_IMAGES=-1
BEST_POINTS=-1
FOUND_ANY=0

for d in "${SPARSE_ROOT}"/*; do
    if [[ ! -d "${d}" ]]; then
        continue
    fi

    # Must contain a COLMAP model
    if [[ ! -f "${d}/cameras.bin" && ! -f "${d}/cameras.txt" ]]; then
        continue
    fi

    FOUND_ANY=1
    MODEL_NAME="$(basename "${d}")"
    TXT_DIR="${TMP_ROOT}/${MODEL_NAME}_txt"

    rm -rf "${TXT_DIR}"
    mkdir -p "${TXT_DIR}"

    echo "--------------------------------------------------"
    echo "Model dir: ${d}"

    # Convert model to TXT
    if ! colmap model_converter \
        --input_path "${d}" \
        --output_path "${TXT_DIR}" \
        --output_type TXT >/dev/null 2>&1; then
        echo "model_converter failed"
        echo
        continue
    fi

    if [[ ! -f "${TXT_DIR}/images.txt" || ! -f "${TXT_DIR}/points3D.txt" ]]; then
        echo "converted TXT files missing"
        echo
        continue
    fi

    # Count registered images:
    # images.txt has 2 lines per image after comments:
    # odd line = image pose/meta
    # even line = points2D
    REG_IMAGES="$(grep -v '^#' "${TXT_DIR}/images.txt" | awk 'NR % 2 == 1' | wc -l | tr -d ' ')"
    POINTS_3D="$(grep -v '^#' "${TXT_DIR}/points3D.txt" | wc -l | tr -d ' ')"

    echo "Registered images : ${REG_IMAGES}"
    echo "Sparse points3D   : ${POINTS_3D}"
    echo

    if [[ "${REG_IMAGES}" -gt "${BEST_REG_IMAGES}" ]]; then
        BEST_REG_IMAGES="${REG_IMAGES}"
        BEST_POINTS="${POINTS_3D}"
        BEST_DIR="${d}"
    elif [[ "${REG_IMAGES}" -eq "${BEST_REG_IMAGES}" && "${POINTS_3D}" -gt "${BEST_POINTS}" ]]; then
        BEST_REG_IMAGES="${REG_IMAGES}"
        BEST_POINTS="${POINTS_3D}"
        BEST_DIR="${d}"
    fi
done

if [[ "${FOUND_ANY}" -eq 0 ]]; then
    echo "No valid sparse models found under ${SPARSE_ROOT}"
    exit 1
fi

echo "=================================================="
echo "Best model selected:"
echo "  BEST_DIR        : ${BEST_DIR}"
echo "  Registered imgs : ${BEST_REG_IMAGES}"
echo "  Points          : ${BEST_POINTS}"
echo "=================================================="
echo

if [[ -z "${BEST_DIR}" ]]; then
    echo "No usable model selected."
    exit 1
fi

if [[ "${SAVE_BEST_TXT}" == "1" ]]; then
    rm -rf "${BEST_TXT_DIR}"
    mkdir -p "${BEST_TXT_DIR}"

    echo "[1/2] Convert best model to TXT"
    colmap model_converter \
        --input_path "${BEST_DIR}" \
        --output_path "${BEST_TXT_DIR}" \
        --output_type TXT >/dev/null 2>&1

    echo
    echo "[2/2] Preview TXT files"
    echo
    echo "== cameras.txt (head) =="
    head -20 "${BEST_TXT_DIR}/cameras.txt" || true

    echo
    echo "== images.txt (head) =="
    head -40 "${BEST_TXT_DIR}/images.txt" || true

    echo
    echo "== points3D.txt (head) =="
    head -20 "${BEST_TXT_DIR}/points3D.txt" || true

    echo
    echo "TXT model saved to: ${BEST_TXT_DIR}"
fi