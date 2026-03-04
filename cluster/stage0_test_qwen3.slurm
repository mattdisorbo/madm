#!/bin/bash
#SBATCH -J madm-s0-test
#SBATCH -o logs/stage0_test_%A_%a.out
#SBATCH -e logs/stage0_test_%A_%a.err
#SBATCH -p mi3001x
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH --array=0-3

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

MODELS=("Qwen/Qwen3-1.7B" "Qwen/Qwen3-4B" "Qwen/Qwen3-8B" "Qwen/Qwen3-14B")
MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"

export N_SAMPLES_BASE=3
export N_SAMPLES_ADVERSARIAL=3

echo "=== Running all datasets with $MODEL (n=3 test) ==="

for script in run_FEVEROUS run_LendingClub run_MoralMachine run_MovieLens run_WikipediaToxicity run_HotelBookings run_Uber run_aime; do
    echo "--- Starting $script ---"
    python scripts/${script}.py "$MODEL"
    echo "--- Finished $script ---"
done

echo "=== All datasets complete ==="
