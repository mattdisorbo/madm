#!/bin/bash
#SBATCH -J madm-sft-eval
#SBATCH -o logs/sft_eval.%j.out
#SBATCH -e logs/sft_eval.%j.err
#SBATCH -p mi2508x
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00

cd "${WORK}/madm"
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export HF_HOME=/work1/harangju/harang/.cache/huggingface
export UV_CACHE_DIR=/work1/harangju/harang/.cache/uv
export PIP_CACHE_DIR=/work1/harangju/harang/.cache/pip
export TMPDIR=/work1/harangju/harang/.cache/tmp
mkdir -p "$UV_CACHE_DIR" "$PIP_CACHE_DIR" "$TMPDIR"
export CUDA_VISIBLE_DEVICES=0
export TOGETHER_API_KEY=unused

ADAPTER="data/together_hotel/adapter"

echo "========== SFT Eval (with signal) at $(date) =========="
for ds in HotelBookings LendingClub WikipediaToxicity MovieLens; do
    echo "---------- EVAL: $ds ----------"
    python scripts/dpo_eval_study3.py --adapter "$ADAPTER" --dataset $ds --n 10
done

echo "========== SFT Eval (no signal / nohint) at $(date) =========="
for ds in HotelBookings LendingClub; do
    echo "---------- EVAL (nohint): $ds ----------"
    python scripts/dpo_eval_study3.py --adapter "$ADAPTER" --dataset $ds --n 10 --nohint
done

echo "ALL DONE at $(date)"
