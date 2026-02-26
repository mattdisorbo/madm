#!/bin/bash
#SBATCH -J madm-stage2
#SBATCH -o logs/stage2_steering.%j.out
#SBATCH -e logs/stage2_steering.%j.err
#SBATCH -p mi3001x
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 04:00:00

cd "${WORK}/madm"
export PYTHONUNBUFFERED=1
export HF_HOME=/work1/harangju/harang/.cache/huggingface
export UV_CACHE_DIR=/work1/harangju/harang/.cache/uv
export PIP_CACHE_DIR=/work1/harangju/harang/.cache/pip

source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
python scripts/collect_stage2_steering.py
