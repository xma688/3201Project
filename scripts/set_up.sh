#!/usr/bin/env bash
set -e

ENV_NAME=3dgs
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_DIR="${PROJECT_ROOT}/gaussian-splatting"

echo "==> Creating conda environment"
conda create -n ${ENV_NAME} python=3.10 -y

echo "==> Activating environment"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ${ENV_NAME}

echo "==> Installing PyTorch"
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.4 -c pytorch -c nvidia -y

echo "==> Fixing MKL / OpenMP compatibility"
conda install -c conda-forge "mkl<2025" "intel-openmp<2025" -y

echo "==> Installing base dependencies"
conda install -c conda-forge plyfile tqdm ninja -y
python -m pip install -U pip setuptools wheel
python -m pip install opencv-python joblib

echo "==> Installing local CUDA extensions"
cd ${PROJECT_DIR}
python -m pip install --no-build-isolation ./submodules/diff-gaussian-rasterization
python -m pip install --no-build-isolation ./submodules/simple-knn
python -m pip install --no-build-isolation ./submodules/fused-ssim

echo "==> Verifying installation"
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -c "import diff_gaussian_rasterization; import simple_knn; import fused_ssim; print('All good.')"

echo "==> Done"
