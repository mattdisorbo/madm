#!/bin/bash
#SBATCH -J madm-v2-sc-q3_4b
#SBATCH -o logs/stage0_v2_sc_q3_4b_%j.out
#SBATCH -e logs/stage0_v2_sc_q3_4b_%j.err
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

MODEL="Qwen/Qwen3-4B"

export N_SAMPLES_CONTROL=0
export N_SAMPLES_CONTROL_COT=0
export N_SAMPLES_SELFCRITIC=100
export N_SAMPLES_CONFIDENCE=0
export N_SAMPLES_COUNTERFACTUAL=0
export N_SAMPLES_EVIDENCE=0

echo "=== v2 selfcritic q3_4b (3 datasets) ==="

echo "--- Starting run_v2_aime ---"
python scripts/run_v2_aime.py "$MODEL"
echo "--- Finished run_v2_aime ---"
echo "--- Starting run_v2_LendingClub ---"
python scripts/run_v2_LendingClub.py "$MODEL"
echo "--- Finished run_v2_LendingClub ---"
echo "--- Starting run_v2_MovieLens ---"
python scripts/run_v2_MovieLens.py "$MODEL"
echo "--- Finished run_v2_MovieLens ---"

echo "=== Done ==="
