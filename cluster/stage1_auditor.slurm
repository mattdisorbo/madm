#!/bin/bash
#SBATCH -J madm-stage1
#SBATCH -o logs/stage1_auditor.%j.out
#SBATCH -e logs/stage1_auditor.%j.err
#SBATCH -p mi2101x
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 04:00:00

cd "${WORK}/madm"
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export HF_HOME=/work1/harangju/harang/.cache/huggingface

python scripts/collect_stage1_auditor.py
