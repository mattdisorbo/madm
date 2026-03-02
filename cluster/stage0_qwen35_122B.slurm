#!/bin/bash
#SBATCH -J madm-s0-122B
#SBATCH -o logs/stage0_122B.%j.out
#SBATCH -e logs/stage0_122B.%j.err
#SBATCH -p mi3001x
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 12:00:00

cd "${WORK}/madm"
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export HF_HOME=/work1/harangju/harang/.cache/huggingface
export UV_CACHE_DIR=/work1/harangju/harang/.cache/uv
export PIP_CACHE_DIR=/work1/harangju/harang/.cache/pip
export TMPDIR=/work1/harangju/harang/.cache/tmp
mkdir -p "$UV_CACHE_DIR" "$PIP_CACHE_DIR" "$TMPDIR"

uv pip install --upgrade transformers
uv pip install --reinstall torch --index-url https://download.pytorch.org/whl/rocm7.1

MODEL="Qwen/Qwen3.5-122B-A10B"
echo "=== Running all datasets with $MODEL ==="

for script in run_aime run_FEVEROUS run_JFLEG run_LendingClub run_MoralMachine run_MovieLens run_WikipediaToxicity run_HotelBookings run_Uber; do
    echo "--- Starting $script ---"
    python scripts/${script}.py "$MODEL"
    echo "--- Finished $script ---"
done

echo "=== All datasets complete ==="
