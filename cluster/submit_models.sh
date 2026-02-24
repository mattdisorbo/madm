#!/usr/bin/env bash
set -euo pipefail

# Submit one job per model for a given dataset.
#
# Usage:
#   bash cluster/submit_models.sh <dataset> <n_samples>
#
# Example:
#   bash cluster/submit_models.sh WikipediaToxicity 10
#
# Each job runs a single model and produces exactly <n_samples> rows per method.

if [ $# -lt 2 ]; then
    echo "Usage: bash cluster/submit_models.sh <dataset> <n_samples>"
    exit 1
fi

DATASET="$1"
N="$2"
REMOTE="amd:\$WORK/madm/"

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
