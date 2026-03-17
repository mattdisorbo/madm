#!/bin/bash
#SBATCH -J madm-study3-think
#SBATCH -o logs/study3_think.%j.out
#SBATCH -e logs/study3_think.%j.err
#SBATCH -p mi2508x
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 04:00:00

cd "${WORK}/madm"
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export HF_HOME=/work1/harangju/harang/.cache/huggingface
export UV_CACHE_DIR=/work1/harangju/harang/.cache/uv
export PIP_CACHE_DIR=/work1/harangju/harang/.cache/pip
export TMPDIR=/work1/harangju/harang/.cache/tmp
mkdir -p "$UV_CACHE_DIR" "$PIP_CACHE_DIR" "$TMPDIR"

export PROVIDER=local
export MODEL="Qwen/Qwen3.5-9B"
export THINKING=1
export N_PER_CONDITION=50
export WORKERS=1

for ds in HotelBookings LendingClub MoralMachine WikipediaToxicity MovieLens; do
    echo "========== Starting $ds at $(date) =========="
    DATASET=$ds python scripts/study3.py
    echo "========== Finished $ds at $(date) =========="
done
echo "ALL DONE at $(date)"
