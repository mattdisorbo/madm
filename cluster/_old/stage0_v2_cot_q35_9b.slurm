#!/bin/bash
#SBATCH -J madm-v2-cot-q35-9b
#SBATCH -o logs/stage0_v2_cot_q35_9b_%j.out
#SBATCH -e logs/stage0_v2_cot_q35_9b_%j.err
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

export N_SAMPLES_CONTROL=0
export N_SAMPLES_CONTROL_COT=100
export N_SAMPLES_SELFCRITIC=0
export N_SAMPLES_CONFIDENCE=0
export N_SAMPLES_COUNTERFACTUAL=0
export N_SAMPLES_EVIDENCE=0

echo "=== v2 control_cot Qwen3.5-9B (all 8 datasets) ==="

for script in run_v2_FEVEROUS run_v2_LendingClub run_v2_MoralMachine run_v2_MovieLens run_v2_WikipediaToxicity run_v2_HotelBookings run_v2_Uber run_v2_aime; do
    echo "--- Starting $script ---"
    python scripts/${script}.py "$MODEL"
    echo "--- Finished $script ---"
done

echo "=== Done ==="
