#!/usr/bin/env bash
set -euo pipefail

# Submit one job per Qwen3 model, each running all 7 settings sequentially.
# Excludes AIME and MoralMachine.
# Each job runs with n=100 samples per method.

N=100
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
    JOB_NAME="madm-${MODEL_SHORT}"
    OUT="logs/${MODEL_SHORT}.%j.out"
    ERR="logs/${MODEL_SHORT}.%j.err"
    echo "==> Submitting ${JOB_NAME} (n=${N})"
    ssh amd "cd \$WORK/madm && MODEL=${MODEL} N=${N} sbatch \
        --partition=mi2101x \
        --job-name='${JOB_NAME}' \
        --output='${OUT}' \
        --error='${ERR}' \
        cluster/run_qwen.slurm"
done
