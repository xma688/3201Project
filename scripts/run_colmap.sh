#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/run_colmap.sh re10k
#   bash scripts/run_colmap.sh dl3dv
#   bash scripts/run_colmap.sh waymo_front
#
# Optional:
#   bash scripts/run_colmap.sh re10k sequential 1
#   bash scripts/run_colmap.sh dl3dv exhaustive 0
#
# Args:
#   $1 = dataset key: re10k | dl3dv | waymo_front
#   $2 = matcher: sequential | exhaustive   (default: sequential)
#   $3 = use_gpu: 1 | 0                     (default: 1)

DATASET_KEY="${1:-}"
MATCHER="${2:-sequential}"
USE_GPU="${3:-1}"

if [[ -z "${DATASET_KEY}" ]]; then
    echo "Usage: bash scripts/run_colmap.sh [re10k|dl3dv|waymo_front] [sequential|exhaustive] [1|0]"
    exit 1
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${PROJECT_ROOT}/data"
OUTPUT_ROOT="${PROJECT_ROOT}/outputs/colmap"

case "${DATASET_KEY}" in
    re10k)
        SCENE_NAME="Re10k-1"
        IMG_DIR="${DATA_ROOT}/Re10k-1/images"
        SINGLE_CAMERA="1"
        ;;
    dl3dv)
        SCENE_NAME="DL3DV-2"
        IMG_DIR="${DATA_ROOT}/DL3DV-2/rgb"
        SINGLE_CAMERA="1"
        ;;
    waymo_front)
        SCENE_NAME="405841_FRONT"
        IMG_DIR="${DATA_ROOT}/405841/FRONT/rgb"
        SINGLE_CAMERA="1"
        ;;
    *)
        echo "Unknown dataset key: ${DATASET_KEY}"
        echo "Allowed: re10k | dl3dv | waymo_front"
        exit 1
        ;;
esac

OUT_DIR="${OUTPUT_ROOT}/${SCENE_NAME}"
DB_PATH="${OUT_DIR}/database.db"
SPARSE_DIR="${OUT_DIR}/sparse"
TXT_DIR="${OUT_DIR}/sparse_txt"
LOG_DIR="${OUT_DIR}/logs"

mkdir -p "${OUT_DIR}" "${SPARSE_DIR}" "${TXT_DIR}" "${LOG_DIR}"

if [[ ! -d "${IMG_DIR}" ]]; then
    echo "Image directory not found: ${IMG_DIR}"
    exit 1
fi

echo "========================================"
echo "DATASET_KEY : ${DATASET_KEY}"
echo "SCENE_NAME  : ${SCENE_NAME}"
echo "IMG_DIR     : ${IMG_DIR}"
echo "OUT_DIR     : ${OUT_DIR}"
echo "MATCHER     : ${MATCHER}"
echo "USE_GPU     : ${USE_GPU}"
echo "========================================"

echo "[0/4] Clean old sparse model index-0 if exists"
mkdir -p "${SPARSE_DIR}"

echo "[1/4] Feature extraction"
colmap feature_extractor \
    --database_path "${DB_PATH}" \
    --image_path "${IMG_DIR}" \
    --ImageReader.single_camera "${SINGLE_CAMERA}" \
    --FeatureExtraction.use_gpu "${USE_GPU}" \
    2>&1 | tee "${LOG_DIR}/feature_extractor.log"

echo "[2/4] Feature matching"
if [[ "${MATCHER}" == "sequential" ]]; then
    colmap sequential_matcher \
        --database_path "${DB_PATH}" \
        --FeatureMatching.use_gpu "${USE_GPU}" \
        2>&1 | tee "${LOG_DIR}/matcher.log"
elif [[ "${MATCHER}" == "exhaustive" ]]; then
    colmap exhaustive_matcher \
        --database_path "${DB_PATH}" \
        --FeatureMatching.use_gpu "${USE_GPU}" \
        2>&1 | tee "${LOG_DIR}/matcher.log"
else
    echo "Unknown matcher: ${MATCHER}"
    echo "Allowed: sequential | exhaustive"
    exit 1
fi

echo "[3/4] Sparse reconstruction (mapper)"
colmap mapper \
    --database_path "${DB_PATH}" \
    --image_path "${IMG_DIR}" \
    --output_path "${SPARSE_DIR}" \
    2>&1 | tee "${LOG_DIR}/mapper.log"

if [[ ! -d "${SPARSE_DIR}/0" ]]; then
    echo "COLMAP did not produce ${SPARSE_DIR}/0"
    echo "Please check logs in ${LOG_DIR}"
    exit 1
fi

echo "[4/4] Convert model to TXT for inspection"
colmap model_converter \
    --input_path "${SPARSE_DIR}/0" \
    --output_path "${TXT_DIR}" \
    --output_type TXT \
    2>&1 | tee "${LOG_DIR}/model_converter.log"

echo "Done."
echo "Binary model : ${SPARSE_DIR}/0"
echo "Text model   : ${TXT_DIR}"