#!/bin/bash
#SBATCH -J madm-stage2
#SBATCH -o logs/stage2_steering.%j.out
#SBATCH -e logs/stage2_steering.%j.err
#SBATCH -p mi3001x
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 04:00:00

set -x  # Enable debugging
set -e  # Exit on error

echo "Starting job..."
cd "${WORK}/madm" || exit 1
echo "Changed to directory: $(pwd)"

export PYTHONUNBUFFERED=1
export HF_HOME=/work1/harangju/harang/.cache/huggingface
export UV_CACHE_DIR=/work1/harangju/harang/.cache/uv
export PIP_CACHE_DIR=/work1/harangju/harang/.cache/pip

echo "Syncing dependencies..."
uv sync

echo "Uninstalling torch..."
uv pip uninstall -y torch torchvision torchaudio || true

echo "Installing CPU-only torch..."
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

echo "Running script..."
uv run python scripts/collect_stage2_steering.py
