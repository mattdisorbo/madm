#!/usr/bin/env bash
set -euo pipefail

# Submit one job per model, each running AIME and JFLEG sequentially.
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
rsync -av --exclude .venv --exclude outputs --exclude .git . "${REMOTE}" || true

echo "==> Ensuring logs directory exists"
ssh amd "mkdir -p \$WORK/madm/logs"

for MODEL in "${MODELS[@]}"; do
    MODEL_SHORT="${MODEL##*/}"
    JOB_NAME="madm-aime-jfleg-${MODEL_SHORT}"
    OUT="logs/aime_jfleg_${MODEL_SHORT}.%j.out"
    ERR="logs/aime_jfleg_${MODEL_SHORT}.%j.err"
    echo "==> Submitting ${JOB_NAME} (n=${N})"
    ssh amd "cd \$WORK/madm && sbatch \
        --job-name='${JOB_NAME}' \
        --output='${OUT}' \
        --error='${ERR}' \
        --export=ALL,MODEL=${MODEL},N=${N} \
        cluster/run_aime_jfleg.slurm"
done

echo "==> Submitted ${#MODELS[@]} jobs"
