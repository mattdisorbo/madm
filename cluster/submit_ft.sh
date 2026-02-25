#!/usr/bin/env bash
set -euo pipefail

# Submit fine-tuned model jobs for LendingClub and HotelBookings.
#
# Usage:
#   bash cluster/submit_ft.sh <n_samples>
#
# Example:
#   bash cluster/submit_ft.sh 150

if [ $# -lt 1 ]; then
    echo "Usage: bash cluster/submit_ft.sh <n_samples>"
    exit 1
fi

N="$1"
REMOTE="amd:\$WORK/madm/"

declare -A FT_MODELS
FT_MODELS["LendingClub"]="ft:gpt-4o-mini-2024-07-18:mit-ide:lendingclub:DDBReFWb"
FT_MODELS["HotelBookings"]="ft:gpt-4o-mini-2024-07-18:mit-ide:hotelbookings:DDBQuoVq"

echo "==> Syncing code to cluster"
rsync -av --exclude .venv --exclude outputs --exclude .git --exclude data . "${REMOTE}"

echo "==> Ensuring logs directory exists"
ssh amd "mkdir -p \$WORK/madm/logs"

for DATASET in "${!FT_MODELS[@]}"; do
    MODEL="${FT_MODELS[$DATASET]}"
    MODEL_SHORT="${MODEL##*:}"
    JOB_NAME="madm-ft-${DATASET}-${MODEL_SHORT}"
    OUT="logs/ft_${DATASET}_${MODEL_SHORT}.%j.out"
    ERR="logs/ft_${DATASET}_${MODEL_SHORT}.%j.err"
    echo "==> Submitting ${JOB_NAME} (n=${N})"
    ssh amd "cd \$WORK/madm && sbatch \
        --job-name='${JOB_NAME}' \
        --output='${OUT}' \
        --error='${ERR}' \
        --export=ALL,DATASET=${DATASET},MODEL=${MODEL},N=${N} \
        cluster/run_model.slurm"
done
