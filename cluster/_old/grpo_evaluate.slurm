#!/bin/bash
#SBATCH -J madm-grpo-eval
#SBATCH -o logs/grpo_evaluate_%j.out
#SBATCH -e logs/grpo_evaluate_%j.err
#SBATCH -p mi2101x
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 08:00:00

cd "${WORK}/madm"
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export HF_HOME=/work1/harangju/harang/.cache/huggingface
export UV_CACHE_DIR=/work1/harangju/harang/.cache/uv
export PIP_CACHE_DIR=/work1/harangju/harang/.cache/pip
export TMPDIR=/work1/harangju/harang/.cache/tmp
mkdir -p "$UV_CACHE_DIR" "$PIP_CACHE_DIR" "$TMPDIR"

uv pip install --reinstall torch --index-url https://download.pytorch.org/whl/rocm7.1
uv pip install --upgrade transformers typing_extensions datasets peft

MODEL="Qwen/Qwen3.5-9B"

echo "=== GRPO Evaluation (baseline) ==="
python scripts/grpo_evaluate.py "$MODEL" --no-lora --max-per-dataset 200
echo "=== Baseline done ==="

echo "=== GRPO Evaluation (LoRA) ==="
python scripts/grpo_evaluate.py "$MODEL" --max-per-dataset 200
echo "=== LoRA done ==="

echo "=== Done ==="
