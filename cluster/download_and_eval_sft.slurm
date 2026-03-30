#!/bin/bash
#SBATCH -J sft-download-eval
#SBATCH -o logs/sft_download_eval.%j.out
#SBATCH -e logs/sft_download_eval.%j.err
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
export CUDA_VISIBLE_DEVICES=0
export TOGETHER_API_KEY=unused

echo "========== Step 1: Download base model at $(date) =========="
python scripts/download_model.py

echo "========== Step 2: SFT Eval (with signal) at $(date) =========="
python scripts/sft_eval.py

echo "========== Step 3: SFT Eval (no signal / ablation) at $(date) =========="
python scripts/sft_eval.py --no-signal

echo "ALL DONE at $(date)"
