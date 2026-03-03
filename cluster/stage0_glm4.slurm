#!/bin/bash
#SBATCH -J madm-s0-glm4
#SBATCH -o logs/stage0_glm4.%j.out
#SBATCH -e logs/stage0_glm4.%j.err
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

uv pip install --reinstall torch --index-url https://download.pytorch.org/whl/rocm7.1
uv pip install --upgrade transformers statsmodels

MODEL="THUDM/glm-4-9b-chat-hf"
echo "=== Running all datasets with $MODEL ==="

for script in run_FEVEROUS run_JFLEG run_LendingClub run_MoralMachine run_MovieLens run_WikipediaToxicity run_HotelBookings run_Uber run_aime; do
    echo "--- Starting $script ---"
    python scripts/${script}.py "$MODEL"
    echo "--- Finished $script ---"
done

echo "=== All datasets complete ==="
