#!/bin/bash
#SBATCH -J madm-s2-1.7b
#SBATCH -o logs/stage2_qwen3_1.7b_%j.out
#SBATCH -e logs/stage2_qwen3_1.7b_%j.err
#SBATCH -p mi2101x
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

uv pip install --reinstall torch --index-url https://download.pytorch.org/whl/rocm7.1
uv pip install --upgrade transformers

export N_SAMPLES=50

echo "=== Stage 2 activation patching with Qwen/Qwen3-1.7B (n=$N_SAMPLES) ==="
python scripts/stage2_attention_patching.py "Qwen/Qwen3-1.7B"
echo "=== Done ==="
