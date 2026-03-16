#!/bin/bash
#SBATCH -J madm-s0-rerun
#SBATCH -o logs/stage0_rerun.%j.out
#SBATCH -e logs/stage0_rerun.%j.err
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
uv pip install --upgrade transformers statsmodels
python -c "import torch; print(f'torch {torch.__version__}')"

# Re-run MovieLens combos that were deleted but never re-ran
# (LendingClub 1.7B already fixed in job 275993)

echo "=== Re-running MovieLens 1.7B (base + ols) ==="
python scripts/run_MovieLens.py "Qwen/Qwen3-1.7B"

echo "=== Re-running MovieLens 4B (base) ==="
python scripts/run_MovieLens.py "Qwen/Qwen3-4B"

echo "=== All re-runs complete ==="
