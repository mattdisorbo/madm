#!/bin/bash
#SBATCH -J madm-4o-mini-base
#SBATCH -o logs/run_4o_mini_base.%j.out
#SBATCH -e logs/run_4o_mini_base.%j.err
#SBATCH -p mi2104x
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00

cd "${WORK}/madm"
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export HF_HOME=/work1/harangju/harang/.cache/huggingface

MODEL="gpt-4o-mini"
N=150

python scripts/run_LendingClub.py   --model "$MODEL" --n $N
python scripts/run_HotelBookings.py --model "$MODEL" --n $N
