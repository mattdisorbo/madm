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

uv sync
uv pip install --reinstall torch --index-url https://download.pytorch.org/whl/rocm7.1
uv pip install --force-reinstall --no-deps transformers

uv run python scripts/collect_stage2_steering.py
