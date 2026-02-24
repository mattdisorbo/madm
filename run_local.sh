#!/bin/bash
# Local run: Qwen3-1.7B and 4B across 7 datasets, n=100 per model x method
# Setup: uv sync (do NOT reinstall ROCm torch)
# Run: bash run_local.sh

set -e

SCRIPTS="scripts"
MODELS=("Qwen/Qwen3-1.7B" "Qwen/Qwen3-4B")
N=100

DATASETS=(
    "run_MovieLens.py"
    "run_LendingClub.py"
    "run_HotelBookings.py"
    "run_Uber.py"
    "run_JFLEG.py"
    "run_WikipediaToxicity.py"
    "run_FEVEROUS.py"
)

for model in "${MODELS[@]}"; do
    for script in "${DATASETS[@]}"; do
        echo "=== $script | $model | n=$N ==="
        uv run python "$SCRIPTS/$script" --model "$model" --n $N
    done
done
