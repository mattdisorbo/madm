#!/bin/bash
#SBATCH -J madm-smoke
#SBATCH -o logs/smoke_test.%j.out
#SBATCH -e logs/smoke_test.%j.err
#SBATCH -p mi2104x
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:30:00

cd "${WORK}/madm"
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export HF_HOME=/work1/harangju/harang/.cache/huggingface
export PYTORCH_ALLOC_CONF=expandable_segments:True

python scripts/smoke_test.py
