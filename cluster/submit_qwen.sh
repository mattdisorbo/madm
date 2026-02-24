#!/usr/bin/env bash
set -euo pipefail

# Submit one job per (dataset, model) for all 4 Qwen3 models across 7 settings.
# Excludes AIME and MoralMachine.
# Each job runs with n=100 samples per method.

N=100
REMOTE="amd:\$WORK/madm/"

DATASETS=(
    "WikipediaToxicity"
    "MovieLens"
    "LendingClub"
    "FEVEROUS"
    "JFLEG"
    "HotelBookings"
    "Uber"
)

MODELS=(
    "Qwen/Qwen3-1.7B"
    "Qwen/Qwen3-4B"
    "Qwen/Qwen3-8B"
    "Qwen/Qwen3-14B"
)

echo "==> Syncing code to cluster"
rsync -av --exclude .venv --exclude outputs --exclude .git . "${REMOTE}"

echo "==> Ensuring logs directory exists"
ssh amd "mkdir -p \$WORK/madm/logs"

for DATASET in "${DATASETS[@]}"; do
    for MODEL in "${MODELS[@]}"; do
        MODEL_SHORT="${MODEL##*/}"
        JOB_NAME="madm-${DATASET}-${MODEL_SHORT}"
        OUT="logs/${DATASET}_${MODEL_SHORT}.%j.out"
        ERR="logs/${DATASET}_${MODEL_SHORT}.%j.err"
        echo "==> Submitting ${JOB_NAME} (n=${N})"
        ssh amd "cd \$WORK/madm && sbatch \
            --job-name='${JOB_NAME}' \
            --output='${OUT}' \
            --error='${ERR}' \
            --time=04:00:00 \
            --export=ALL,DATASET=${DATASET},MODEL=${MODEL},N=${N} \
            cluster/run_model.slurm"
    done
done
