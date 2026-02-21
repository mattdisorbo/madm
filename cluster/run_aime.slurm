#!/bin/bash
#SBATCH -J madm-aime
#SBATCH -o logs/run_aime.%j.out
#SBATCH -e logs/run_aime.%j.err
#SBATCH -p mi2104x
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:30:00

cd "${WORK}/madm"
source .venv/bin/activate
export PYTHONUNBUFFERED=1

python scripts/run_aime.py
