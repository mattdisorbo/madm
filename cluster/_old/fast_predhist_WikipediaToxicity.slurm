#!/bin/bash
#SBATCH -J madm-ph-wikipediatoxicity
#SBATCH -o logs/fast_predhist_wikipediatoxicity_%j.out
#SBATCH -e logs/fast_predhist_wikipediatoxicity_%j.err
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
uv pip install --upgrade transformers typing_extensions scikit-learn scipy markupsafe jinja2

MODEL="Qwen/Qwen3.5-9B"

export N_SAMPLES_CONTROL=0
export N_SAMPLES_CONTROL_COT=0
export N_SAMPLES_SELFCRITIC=0
export N_SAMPLES_CONFIDENCE=0
export N_SAMPLES_COUNTERFACTUAL=0
export N_SAMPLES_COUNTERFACTUAL2=0
export N_SAMPLES_EVIDENCE=0
export N_SAMPLES_PRED_REASONING=100
export N_SAMPLES_HISTORY=100
export N_OAI=0
export N_NANO=0
export N_QWEN=1

echo "--- Starting WikipediaToxicity (prediction_reasoning, history) ---"
python scripts/run_v2_WikipediaToxicity.py "$MODEL"
echo "--- Finished WikipediaToxicity ---"
echo "=== Done ==="
