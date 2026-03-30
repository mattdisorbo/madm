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
export HF_HUB_OFFLINE=1
export TOGETHER_API_KEY=unused
export UV_CACHE_DIR=/work1/harangju/harang/.cache/uv
export PIP_CACHE_DIR=/work1/harangju/harang/.cache/pip
export TMPDIR=/work1/harangju/harang/.cache/tmp
mkdir -p "$UV_CACHE_DIR" "$PIP_CACHE_DIR" "$TMPDIR"
export CUDA_VISIBLE_DEVICES=0

echo "========== SFT Eval (with signal) at $(date) =========="
python scripts/sft_eval.py

echo "========== SFT Eval (no signal / ablation) at $(date) =========="
python scripts/sft_eval.py --no-signal

echo "ALL DONE at $(date)"
