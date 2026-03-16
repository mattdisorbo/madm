#!/bin/bash
#SBATCH -J madm-grpo-hotel
#SBATCH -o logs/grpo_train_hotel_%j.out
#SBATCH -e logs/grpo_train_hotel_%j.err
#SBATCH -p mi2101x
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 12:00:00

set -e
cd "${WORK}/madm"
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export HF_HOME=/work1/harangju/harang/.cache/huggingface
export UV_CACHE_DIR=/work1/harangju/harang/.cache/uv
export PIP_CACHE_DIR=/work1/harangju/harang/.cache/pip
export TMPDIR=/work1/harangju/harang/.cache/tmp
mkdir -p "$UV_CACHE_DIR" "$PIP_CACHE_DIR" "$TMPDIR" logs/
export TORCH_BLAS_PREFER_HIPBLASLT=0

uv pip install --reinstall torch --index-url https://download.pytorch.org/whl/rocm7.1
uv pip install --upgrade transformers typing_extensions datasets trl peft accelerate

MODEL="Qwen/Qwen3-8B"

echo "=== GRPO Training (HotelBookings) ==="
python scripts/grpo_train_hotel.py "$MODEL"
echo "=== Training done ==="

echo "=== Baseline Evaluation ==="
python scripts/grpo_eval_hotel.py "$MODEL" --no-lora --n-per-condition 50
echo "=== Baseline done ==="

echo "=== GRPO Evaluation ==="
python scripts/grpo_eval_hotel.py "$MODEL" --n-per-condition 50
echo "=== GRPO done ==="

echo "=== All done ==="
