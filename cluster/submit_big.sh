#!/usr/bin/env bash
set -euo pipefail

# Submit one job per model, each running all 9 datasets sequentially.
# MoralMachine and AIME run last.
# n=150 per method.

N=150
REMOTE="amd:\$WORK/madm/"

MODELS=(
    "gpt-5-mini-2025-08-07"
    "gpt-5-nano-2025-08-07"
    "Qwen/Qwen3-1.7B"
    "Qwen/Qwen3-4B"
    "Qwen/Qwen3-8B"
    "Qwen/Qwen3-14B"
    "THUDM/glm-4-9b-chat-hf"
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
    ssh amd "cd \$WORK/madm && sbatch \
        --job-name='${JOB_NAME}' \
        --output='${OUT}' \
        --error='${ERR}' \
        --export=ALL,MODEL=${MODEL},N=${N} \
        cluster/run_big.slurm"
done

echo "==> Submitted ${#MODELS[@]} jobs"
