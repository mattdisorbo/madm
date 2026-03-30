#!/bin/bash
#SBATCH -J sft-train-eval
#SBATCH -o logs/sft_train_eval.%j.out
#SBATCH -e logs/sft_train_eval.%j.err
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

echo "========== SFT Training at $(date) =========="
python scripts/sft_escalation.py --output-dir outputs/sft_esc --holdout MovieLens --n 5

echo "========== SFT Eval (per format) at $(date) =========="
for fmt in original dollar wording; do
    for ds in HotelBookings LendingClub WikipediaToxicity MovieLens; do
        echo "---------- EVAL: $ds ($fmt) ----------"
        python scripts/dpo_eval_study3.py --adapter outputs/sft_esc --dataset $ds --n 5 --formats $fmt
    done
done

echo "========== SFT Eval (nohint ablation) at $(date) =========="
for ds in HotelBookings LendingClub; do
    echo "---------- EVAL (nohint): $ds ----------"
    python scripts/dpo_eval_study3.py --adapter outputs/sft_esc --dataset $ds --n 5 --formats original --nohint
done

echo "ALL DONE at $(date)"
