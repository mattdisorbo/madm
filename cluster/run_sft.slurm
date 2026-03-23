#!/bin/bash
#SBATCH -J sft-esc
#SBATCH -o logs/sft_esc.%j.out
#SBATCH -e logs/sft_esc.%j.err
#SBATCH -p mi2508x
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 10:00:00

cd "${WORK}/madm"
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export HF_HOME=/work1/harangju/harang/.cache/huggingface
export TOGETHER_API_KEY=unused
export PYTORCH_ALLOC_CONF=expandable_segments:True

python scripts/sft_escalation.py --output-dir outputs/sft_esc --holdout MovieLens --n 5

# Eval on ALL 5 datasets (including holdout)
for ds in HotelBookings LendingClub MoralMachine WikipediaToxicity MovieLens; do
    echo "========== EVAL: $ds =========="
    python scripts/dpo_eval_study3.py --adapter outputs/sft_esc --dataset $ds --n 5 --formats original
done
echo DONE
