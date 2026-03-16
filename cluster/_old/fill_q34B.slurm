#!/bin/bash
#SBATCH -J madm-fill-q34B
#SBATCH -o logs/fill_q34B_%j.out
#SBATCH -e logs/fill_q34B_%j.err
#SBATCH -p mi2101x
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 06:00:00

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

echo "=== Fill missing for Qwen3-4B (8 runs) ==="

export N_SAMPLES_CONTROL=0
export N_SAMPLES_CONTROL_COT=0
export N_SAMPLES_SELFCRITIC=0
export N_SAMPLES_CONFIDENCE=100
export N_SAMPLES_COUNTERFACTUAL=100
export N_SAMPLES_EVIDENCE=0
export N_OAI=0
export N_NANO=0
export N_QWEN=1
echo "--- Starting run_v2_LendingClub (['confidence', 'counterfactual']) ---"
python scripts/run_v2_LendingClub.py "$MODEL"
echo "--- Finished run_v2_LendingClub ---"

export N_SAMPLES_CONTROL=0
export N_SAMPLES_CONTROL_COT=0
export N_SAMPLES_SELFCRITIC=100
export N_SAMPLES_CONFIDENCE=100
export N_SAMPLES_COUNTERFACTUAL=100
export N_SAMPLES_EVIDENCE=0
export N_OAI=0
export N_NANO=0
export N_QWEN=1
echo "--- Starting run_v2_MovieLens (['confidence', 'counterfactual', 'selfcritic']) ---"
python scripts/run_v2_MovieLens.py "$MODEL"
echo "--- Finished run_v2_MovieLens ---"

export N_SAMPLES_CONTROL=0
export N_SAMPLES_CONTROL_COT=0
export N_SAMPLES_SELFCRITIC=0
export N_SAMPLES_CONFIDENCE=100
export N_SAMPLES_COUNTERFACTUAL=100
export N_SAMPLES_EVIDENCE=0
export N_OAI=0
export N_NANO=0
export N_QWEN=1
echo "--- Starting run_v2_Uber (['confidence', 'counterfactual']) ---"
python scripts/run_v2_Uber.py "$MODEL"
echo "--- Finished run_v2_Uber ---"

export N_SAMPLES_CONTROL=0
export N_SAMPLES_CONTROL_COT=0
export N_SAMPLES_SELFCRITIC=0
export N_SAMPLES_CONFIDENCE=0
export N_SAMPLES_COUNTERFACTUAL=100
export N_SAMPLES_EVIDENCE=0
export N_OAI=0
export N_NANO=0
export N_QWEN=1
echo "--- Starting run_v2_WikipediaToxicity (['counterfactual']) ---"
python scripts/run_v2_WikipediaToxicity.py "$MODEL"
echo "--- Finished run_v2_WikipediaToxicity ---"

echo "=== Done ==="
