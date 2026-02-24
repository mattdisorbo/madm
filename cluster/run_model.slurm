#!/bin/bash
#SBATCH -p mi2104x
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00

cd "${WORK}/madm"
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export HF_HOME=/work1/harangju/harang/.cache/huggingface
export PYTORCH_ALLOC_CONF=expandable_segments:True

python scripts/run_${DATASET}.py --model "${MODEL}" --n "${N}"
