#!/usr/bin/env bash
set -euo pipefail

# Submit a job to the AMD cluster from your local machine.
#
# Usage:
#   bash cluster/submit.sh cluster/smoke_test.slurm

if [ $# -eq 0 ]; then
    echo "Usage: bash cluster/submit.sh <slurm-script>"
    exit 1
fi

SCRIPT="$1"
REMOTE="amd:\$WORK/madm/"

echo "==> Syncing code to cluster"
rsync -av --exclude .venv --exclude outputs --exclude .git --exclude data . "${REMOTE}"

echo "==> Ensuring logs directory exists"
ssh amd "mkdir -p \$WORK/madm/logs"

echo "==> Submitting ${SCRIPT}"
ssh amd "cd \$WORK/madm && sbatch ${SCRIPT}"
