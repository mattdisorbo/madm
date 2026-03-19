#!/bin/bash
#SBATCH -J dpo-study3
#SBATCH -o logs/dpo_study3.%j.out
#SBATCH -e logs/dpo_study3.%j.err
#SBATCH -p mi2508x
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 06:00:00

cd "${WORK}/madm"
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export HF_HOME=/work1/harangju/harang/.cache/huggingface
export TOGETHER_API_KEY=unused

for ds in HotelBookings LendingClub MoralMachine WikipediaToxicity MovieLens; do
    echo "========== $ds =========="
    python scripts/dpo_eval_study3.py --adapter outputs/dpo_full --dataset $ds --n 10
done
echo DONE
