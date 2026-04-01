#!/bin/bash
#SBATCH -J sft-train-eval
#SBATCH -o logs/sft_train_eval.%j.out
#SBATCH -e logs/sft_train_eval.%j.err
#SBATCH -p mi2508x
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 04:00:00

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
export HF_HUB_OFFLINE=1

# Training (fast, resumes from checkpoint)
echo "========== SFT Training at $(date) =========="
python scripts/sft_escalation.py --output-dir outputs/sft_esc --holdout MovieLens --n 5

# Eval: format and dataset passed as args
FMT="${1:-original}"
DS="${2:-HotelBookings}"
NOHINT="${3:-}"

if [ -n "$NOHINT" ]; then
    echo "========== SFT Eval: $DS ($FMT, nohint) at $(date) =========="
    python scripts/dpo_eval_study3.py --adapter outputs/sft_esc --dataset $DS --n 5 --formats $FMT --nohint
else
    echo "========== SFT Eval: $DS ($FMT) at $(date) =========="
    python scripts/dpo_eval_study3.py --adapter outputs/sft_esc --dataset $DS --n 5 --formats $FMT
fi

echo "DONE at $(date)"
