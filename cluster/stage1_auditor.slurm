#!/bin/bash
#SBATCH -J madm-stage1
#SBATCH -o logs/stage1_auditor.%j.out
#SBATCH -e logs/stage1_auditor.%j.err
#SBATCH -p mi2104x
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00

cd "${WORK}/madm"
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export PATH="$HOME/.local/bin:$PATH"

# Force reinstall ROCm PyTorch
uv pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/rocm7.1

python scripts/collect_stage1_auditor.py
