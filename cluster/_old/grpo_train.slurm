#!/bin/bash
#SBATCH -J madm-grpo
#SBATCH -o logs/grpo_train_%j.out
#SBATCH -e logs/grpo_train_%j.err
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
mkdir -p "$UV_CACHE_DIR" "$PIP_CACHE_DIR" "$TMPDIR"

uv pip install --reinstall torch --index-url https://download.pytorch.org/whl/rocm7.1
uv pip install --upgrade "transformers @ git+https://github.com/huggingface/transformers.git@main" typing_extensions datasets trl peft accelerate scikit-learn
uv pip install --reinstall torchvision --index-url https://download.pytorch.org/whl/rocm7.1

MODEL="Qwen/Qwen3.5-9B"

echo "=== GRPO Training (LendingClub) ==="
python scripts/grpo_escalation.py "$MODEL"
echo "=== Training done ==="

echo "=== Baseline Evaluation ==="
python scripts/grpo_evaluate.py "$MODEL" --no-lora --n-eval 500
echo "=== Baseline done ==="

echo "=== GRPO Evaluation ==="
python scripts/grpo_evaluate.py "$MODEL" --n-eval 500
echo "=== GRPO done ==="

echo "=== All done ==="
