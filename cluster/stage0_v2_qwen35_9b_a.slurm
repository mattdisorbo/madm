#!/bin/bash
#SBATCH -J madm-v2-q35-9b-a
#SBATCH -o logs/stage0_v2_qwen35_9b_a_%j.out
#SBATCH -e logs/stage0_v2_qwen35_9b_a_%j.err
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
uv pip install --upgrade transformers typing_extensions

MODEL="Qwen/Qwen3.5-9B"

export N_SAMPLES_CONTROL=100
export N_SAMPLES_SELFCRITIC=100
export N_SAMPLES_CONFIDENCE=100
export N_SAMPLES_COUNTERFACTUAL=100
export N_SAMPLES_EVIDENCE=100

echo "=== v2 Qwen3.5-9B batch A (4 datasets) ==="

for script in run_v2_FEVEROUS run_v2_LendingClub run_v2_MoralMachine run_v2_MovieLens; do
    echo "--- Starting $script ---"
    python scripts/${script}.py "$MODEL"
    echo "--- Finished $script ---"
done

echo "=== Done ==="
