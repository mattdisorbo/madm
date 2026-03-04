#!/bin/bash
#SBATCH -J madm-s0-fl
#SBATCH -o logs/stage0_fl_%A_%a.out
#SBATCH -e logs/stage0_fl_%A_%a.err
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
uv pip install --upgrade transformers statsmodels

# Wait for torch to be fully installed
python -c "import torch; print(f'PyTorch {torch.__version__} ready')"

MODELS=("Qwen/Qwen3-1.7B" "Qwen/Qwen3-4B" "Qwen/Qwen3-14B")
MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"

export N_SAMPLES_BASE=100
export N_SAMPLES_ADVERSARIAL=100
export N_SAMPLES_AGREEABLE=0
export N_SAMPLES_MULTIAGENT=0

echo "=== Running FEVEROUS + LendingClub with $MODEL (n=100) ==="

for script in run_FEVEROUS run_LendingClub; do
    echo "--- Starting $script ---"
    python scripts/${script}.py "$MODEL"
    echo "--- Finished $script ---"
done

echo "=== Done ==="
