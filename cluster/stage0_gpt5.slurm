#!/bin/bash
#SBATCH -J madm-s0-gpt5
#SBATCH -o logs/stage0_gpt5.%j.out
#SBATCH -e logs/stage0_gpt5.%j.err
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

uv pip install --upgrade openai

export N_OAI=1
export N_NANO=1
export N_QWEN=0

echo "=== Running GPT-5-mini and GPT-5-nano ==="

for script in run_FEVEROUS run_JFLEG run_LendingClub run_MoralMachine run_MovieLens run_WikipediaToxicity run_HotelBookings run_Uber run_aime; do
    echo "--- Starting $script ---"
    python scripts/${script}.py
    echo "--- Finished $script ---"
done

echo "=== All datasets complete ==="
