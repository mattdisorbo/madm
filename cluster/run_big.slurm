#!/bin/bash
#SBATCH -p mi3001x
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 12:00:00

cd "${WORK}/madm"
source .venv/bin/activate
[ -f ~/.openai_key ] && source ~/.openai_key
export PYTHONUNBUFFERED=1
export HF_HOME=/work1/harangju/harang/.cache/huggingface
export PYTORCH_ALLOC_CONF=expandable_segments:True

for DATASET in WikipediaToxicity MovieLens LendingClub FEVEROUS JFLEG HotelBookings Uber MoralMachine AIME; do
    echo "==> Running ${DATASET} with ${MODEL} (n=${N})"
    python scripts/run_${DATASET}.py --model "${MODEL}" --n "${N}"
done
