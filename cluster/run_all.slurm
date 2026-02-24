#!/bin/bash
#SBATCH -J madm-all
#SBATCH -o logs/run_all.%j.out
#SBATCH -e logs/run_all.%j.err
#SBATCH -p mi3008x
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 12:00:00

cd "${WORK}/madm"
source .venv/bin/activate
[ -f ~/.deepseek_key ] && source ~/.deepseek_key
export PYTHONUNBUFFERED=1
export HF_HOME=/work1/harangju/harang/.cache/huggingface
export PYTORCH_ALLOC_CONF=expandable_segments:True

python scripts/run_all.py
