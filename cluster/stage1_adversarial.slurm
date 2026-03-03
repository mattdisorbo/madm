#!/bin/bash
#SBATCH -J madm-stage1
#SBATCH -o logs/stage1_adversarial.%j.out
#SBATCH -e logs/stage1_adversarial.%j.err
#SBATCH -p mi2101x
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 04:00:00

cd "${WORK}/madm"
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export HF_HOME=/work1/harangju/harang/.cache/huggingface
export UV_CACHE_DIR=/work1/harangju/harang/.cache/uv
export PIP_CACHE_DIR=/work1/harangju/harang/.cache/pip
export TMPDIR=/work1/harangju/harang/.cache/tmp
mkdir -p "$UV_CACHE_DIR" "$PIP_CACHE_DIR" "$TMPDIR"

uv pip install --upgrade transformers
uv pip install --reinstall torch --index-url https://download.pytorch.org/whl/rocm7.1

python scripts/collect_stage1_adversarial.py
