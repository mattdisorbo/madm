#!/bin/bash
#SBATCH -J dpo-v2
#SBATCH -o logs/dpo_v2.%j.out
#SBATCH -e logs/dpo_v2.%j.err
#SBATCH -p mi2508x
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 10:00:00

cd "${WORK}/madm"
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export HF_HOME=/work1/harangju/harang/.cache/huggingface
export TOGETHER_API_KEY=unused

python scripts/dpo_escalation_v2.py --output-dir outputs/dpo_v2 --n 20

# Eval on full study3 prompts
echo "========== EVAL: HotelBookings =========="
python scripts/dpo_eval_study3.py --adapter outputs/dpo_v2 --dataset HotelBookings --n 5

echo "========== EVAL: LendingClub =========="
python scripts/dpo_eval_study3.py --adapter outputs/dpo_v2 --dataset LendingClub --n 5

echo DONE
