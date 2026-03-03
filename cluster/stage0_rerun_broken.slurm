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

uv pip install --upgrade transformers statsmodels
uv pip install --reinstall torch --index-url https://download.pytorch.org/whl/rocm7.1

# Delete bad results and re-run broken combos
# 1.7B: LendingClub base (44/50 pred NaN), MovieLens base+ols (23/50 esc NaN)
# 4B: MovieLens base (50/50 esc NaN)

echo "=== Deleting bad results ==="
rm -f results/LendingClub/base_Qwen3-1.7B.csv
rm -f results/MovieLens/base_Qwen3-1.7B.csv results/MovieLens/ols_Qwen3-1.7B.csv
rm -f results/MovieLens/base_Qwen3-4B.csv

echo "=== Re-running 1.7B broken datasets ==="
python scripts/run_LendingClub.py "Qwen/Qwen3-1.7B"
python scripts/run_MovieLens.py "Qwen/Qwen3-1.7B"

echo "=== Re-running 4B broken datasets ==="
python scripts/run_MovieLens.py "Qwen/Qwen3-4B"

echo "=== All re-runs complete ==="
