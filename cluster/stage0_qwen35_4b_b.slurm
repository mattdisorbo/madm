#!/bin/bash
#SBATCH -J madm-q35-4b-b
#SBATCH -o logs/stage0_qwen35_4b_b_%j.out
#SBATCH -e logs/stage0_qwen35_4b_b_%j.err
#SBATCH -p mi2101x
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 08:00:00

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

MODEL="Qwen/Qwen3.5-4B"

export N_SAMPLES_BASE=100
export N_SAMPLES_ADVERSARIAL=100
export N_SAMPLES_AGREEABLE=100
export N_SAMPLES_MULTIAGENT=100

echo "=== Qwen3.5-4B batch B (4 datasets) ==="

for script in run_WikipediaToxicity run_HotelBookings run_Uber run_aime; do
    echo "--- Starting $script ---"
    python scripts/${script}.py "$MODEL"
    echo "--- Finished $script ---"
done

echo "=== Done ==="
