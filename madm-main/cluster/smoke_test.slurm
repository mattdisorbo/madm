#!/bin/bash
#SBATCH -J madm-smoke
#SBATCH -o logs/smoke_test.%j.out
#SBATCH -e logs/smoke_test.%j.err
#SBATCH -p devel
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:30:00

cd "${WORK}/madm"
source .venv/bin/activate

python scripts/smoke_test.py
