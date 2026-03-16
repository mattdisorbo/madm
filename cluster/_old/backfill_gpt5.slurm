#!/bin/bash
#SBATCH -J madm-bf-gpt5
#SBATCH -o logs/backfill_gpt5_%j.out
#SBATCH -e logs/backfill_gpt5_%j.err
#SBATCH -p mi2101x
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 04:00:00

cd "${WORK}/madm"
source .venv/bin/activate
source ~/.bashrc
export PYTHONUNBUFFERED=1
export UV_CACHE_DIR=/work1/harangju/harang/.cache/uv
export PIP_CACHE_DIR=/work1/harangju/harang/.cache/pip
export TMPDIR=/work1/harangju/harang/.cache/tmp
mkdir -p "$UV_CACHE_DIR" "$PIP_CACHE_DIR" "$TMPDIR"

uv pip install --upgrade openai typing_extensions

export N_QWEN=0

# === AIME ===
# GPT-5-nano: all 4 methods missing
# GPT-5-mini: only multiagent missing
echo "=== AIME: nano all methods, mini multiagent only ==="

# Run nano with all methods
export N_OAI=0
export N_NANO=1
export N_SAMPLES_BASE=100
export N_SAMPLES_ADVERSARIAL=100
export N_SAMPLES_AGREEABLE=100
export N_SAMPLES_MULTIAGENT=100
python scripts/run_aime.py

# Run mini with multiagent only
export N_OAI=1
export N_NANO=0
export N_SAMPLES_BASE=0
export N_SAMPLES_ADVERSARIAL=0
export N_SAMPLES_AGREEABLE=0
export N_SAMPLES_MULTIAGENT=100
python scripts/run_aime.py

# === HotelBookings: both models, all 4 methods ===
echo "=== HotelBookings: both models, all methods ==="
export N_OAI=1
export N_NANO=1
export N_SAMPLES_BASE=100
export N_SAMPLES_ADVERSARIAL=100
export N_SAMPLES_AGREEABLE=100
export N_SAMPLES_MULTIAGENT=100
python scripts/run_HotelBookings.py

# === Uber: both models, all 4 methods ===
echo "=== Uber: both models, all methods ==="
python scripts/run_Uber.py

echo "=== GPT-5 backfill complete ==="
