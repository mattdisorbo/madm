#!/bin/bash
#SBATCH -J madm-s0-q3-8B-fix
#SBATCH -o logs/stage0_q3_8B_fix.%j.out
#SBATCH -e logs/stage0_q3_8B_fix.%j.err
#SBATCH -p mi3001x
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

uv pip install --upgrade transformers statsmodels
uv pip install --reinstall torch --index-url https://download.pytorch.org/whl/rocm7.1

MODEL="Qwen/Qwen3-8B"
echo "=== Running missing datasets with $MODEL ==="

for script in run_LendingClub run_MovieLens; do
    echo "--- Starting $script ---"
    python scripts/${script}.py "$MODEL"
    echo "--- Finished $script ---"
done

echo "=== Fix complete ==="
