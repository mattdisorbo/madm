#!/bin/bash
#SBATCH -J madm-bf-glm4
#SBATCH -o logs/backfill_glm4_%j.out
#SBATCH -e logs/backfill_glm4_%j.err
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

# Missing: all AIME methods + multiagent Uber
export N_SAMPLES_BASE=100
export N_SAMPLES_ADVERSARIAL=100
export N_SAMPLES_AGREEABLE=100
export N_SAMPLES_MULTIAGENT=100

echo "=== GLM-4 backfill: AIME (all methods) + Uber (multiagent) ==="

echo "--- Starting run_aime ---"
python scripts/run_aime.py "$MODEL"
echo "--- Finished run_aime ---"

# For Uber, only need multiagent — set others to 0
export N_SAMPLES_BASE=0
export N_SAMPLES_ADVERSARIAL=0
export N_SAMPLES_AGREEABLE=0
export N_SAMPLES_MULTIAGENT=100

echo "--- Starting run_Uber (multiagent only) ---"
python scripts/run_Uber.py "$MODEL"
echo "--- Finished run_Uber ---"

echo "=== GLM-4 backfill complete ==="
