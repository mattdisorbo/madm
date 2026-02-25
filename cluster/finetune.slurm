#!/bin/bash
#SBATCH -J madm-finetune
#SBATCH -o logs/finetune.%j.out
#SBATCH -e logs/finetune.%j.err
#SBATCH -p devel
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:30:00

cd "${WORK}/madm"
source .venv/bin/activate
export PYTHONUNBUFFERED=1

mkdir -p finetune/data

echo "==> Preparing fine-tuning data"
python scripts/finetune/prepare_data.py

echo "==> Submitting fine-tuning jobs to OpenAI"
python scripts/finetune/submit_jobs.py
