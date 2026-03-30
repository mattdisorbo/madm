#!/bin/bash
#SBATCH -J dpo-v3
#SBATCH -o logs/dpo_v3.%j.out
#SBATCH -e logs/dpo_v3.%j.err
#SBATCH -p mi2508x
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 10:00:00

cd "${WORK}/madm"
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export HF_HOME=/work1/harangju/harang/.cache/huggingface
export UV_CACHE_DIR=/work1/harangju/harang/.cache/uv
export PIP_CACHE_DIR=/work1/harangju/harang/.cache/pip
export TMPDIR=/work1/harangju/harang/.cache/tmp
mkdir -p "$UV_CACHE_DIR" "$PIP_CACHE_DIR" "$TMPDIR"
export TOGETHER_API_KEY=unused
export PYTORCH_ALLOC_CONF=expandable_segments:True

echo "========== DPO TRAINING at $(date) =========="
python scripts/dpo_escalation_v3.py --output-dir outputs/dpo_v3 --holdout MovieLens --n 20

echo "========== DPO EVAL (all datasets, all formats) at $(date) =========="
for ds in HotelBookings LendingClub MoralMachine WikipediaToxicity MovieLens; do
    echo "---------- EVAL: $ds ----------"
    python scripts/dpo_eval_study3.py --adapter outputs/dpo_v3 --dataset $ds --n 10
done

echo "ALL DONE at $(date)"
