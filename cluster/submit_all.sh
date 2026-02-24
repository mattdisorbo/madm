#!/usr/bin/env bash
set -euo pipefail

# Submit one job per dataset × model (36 jobs total).
#
# Usage:
#   bash cluster/submit_all.sh <n_samples>
#
# Example:
#   bash cluster/submit_all.sh 100

if [ $# -lt 1 ]; then
    echo "Usage: bash cluster/submit_all.sh <n_samples>"
    exit 1
fi

N="$1"
REMOTE="amd:\$WORK/madm/"

DATASETS=(
    WikipediaToxicity
    MovieLens
    LendingClub
    FEVEROUS
    JFLEG
    MoralMachine
    HotelBookings
    Uber
    aime
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
            --export=ALL,DATASET=${DATASET},MODEL=${MODEL},N=${N} \
            cluster/run_model.slurm"
    done
done

echo "==> Submitted $((${#DATASETS[@]} * ${#MODELS[@]})) jobs"
