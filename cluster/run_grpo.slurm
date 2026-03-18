#!/bin/bash
#SBATCH -J madm-grpo
#SBATCH -o logs/grpo.%j.out
#SBATCH -e logs/grpo.%j.err
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
export TOGETHER_API_KEY=unused  # study3.py import requires this

# Full run
python scripts/grpo_escalation.py --output-dir outputs/grpo_full

# Evaluate
python scripts/grpo_eval.py --model Qwen/Qwen2.5-1.5B-Instruct --adapter outputs/grpo_full --n 20

echo "DONE at $(date)"
