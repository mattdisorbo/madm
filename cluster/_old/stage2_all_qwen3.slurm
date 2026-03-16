#!/bin/bash
#SBATCH -J madm-s2-all
#SBATCH -o logs/stage2_all_qwen3_%A_%a.out
#SBATCH -e logs/stage2_all_qwen3_%A_%a.err
#SBATCH -p mi2101x
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 04:00:00
#SBATCH --array=0-2

cd "${WORK}/madm"
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export HF_HOME=/work1/harangju/harang/.cache/huggingface
export UV_CACHE_DIR=/work1/harangju/harang/.cache/uv
export PIP_CACHE_DIR=/work1/harangju/harang/.cache/pip
export TMPDIR=/work1/harangju/harang/.cache/tmp
mkdir -p "$UV_CACHE_DIR" "$PIP_CACHE_DIR" "$TMPDIR"

uv pip install --reinstall torch --index-url https://download.pytorch.org/whl/rocm7.1
uv pip install --upgrade transformers

MODELS=("Qwen/Qwen3-1.7B" "Qwen/Qwen3-8B" "Qwen/Qwen3-14B")
MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"

export N_SAMPLES=50

echo "=== Stage 2 activation patching with $MODEL (n=$N_SAMPLES) ==="
python scripts/stage2_attention_patching.py "$MODEL"
echo "=== Done ==="
