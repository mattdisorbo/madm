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
    echo "==> Submitting madm-${MODEL_SHORT} (n=${N})"
    ssh amd "printf '#!/bin/bash\n#SBATCH -p mi2101x\n#SBATCH -N 1\n#SBATCH -n 1\n#SBATCH -t 12:00:00\n#SBATCH --job-name=madm-${MODEL_SHORT}\n#SBATCH --output=logs/${MODEL_SHORT}.%%j.out\n#SBATCH --error=logs/${MODEL_SHORT}.%%j.err\nexport MODEL=${MODEL}\nexport N=${N}\n' > /tmp/run_${MODEL_SHORT}.slurm && tail -n +6 \$WORK/madm/cluster/run_qwen.slurm >> /tmp/run_${MODEL_SHORT}.slurm && cd \$WORK/madm && sbatch --partition=mi2101x /tmp/run_${MODEL_SHORT}.slurm"
done
