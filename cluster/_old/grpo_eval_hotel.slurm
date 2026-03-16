#!/bin/bash
#SBATCH -J madm-grpo-hotel-eval
#SBATCH -o logs/grpo_eval_hotel_%j.out
#SBATCH -e logs/grpo_eval_hotel_%j.err
#SBATCH -p mi2101x
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 08:00:00

set -e
cd "${WORK}/madm"
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export HF_HOME=/work1/harangju/harang/.cache/huggingface
export UV_CACHE_DIR=/work1/harangju/harang/.cache/uv
export PIP_CACHE_DIR=/work1/harangju/harang/.cache/pip
export TMPDIR=/work1/harangju/harang/.cache/tmp
mkdir -p "$UV_CACHE_DIR" "$PIP_CACHE_DIR" "$TMPDIR" logs/

uv pip install --reinstall torch --index-url https://download.pytorch.org/whl/rocm7.1
uv pip install --upgrade "transformers @ git+https://github.com/huggingface/transformers.git@main" typing_extensions datasets peft accelerate
uv pip install --reinstall torchvision --index-url https://download.pytorch.org/whl/rocm7.1

MODEL="Qwen/Qwen3-8B"

echo "=== Baseline Evaluation (all datasets) ==="
python scripts/grpo_eval_hotel.py "$MODEL" --no-lora --n-per-condition 50
echo "=== Baseline done ==="

echo "=== GRPO Evaluation (all datasets) ==="
python scripts/grpo_eval_hotel.py "$MODEL" --n-per-condition 50
echo "=== GRPO done ==="

echo "=== Done ==="
