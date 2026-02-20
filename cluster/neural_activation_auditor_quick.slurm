#!/bin/bash
#SBATCH -J madm-quick
#SBATCH -o logs/neural_activation_auditor_quick.%j.out
#SBATCH -e logs/neural_activation_auditor_quick.%j.err
#SBATCH -p mi2104x
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:30:00

cd "${WORK}/madm"
source .venv/bin/activate
export PYTHONUNBUFFERED=1

python scripts/neural_activation_auditor_quick.py
