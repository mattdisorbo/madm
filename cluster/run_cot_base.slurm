#!/bin/bash
#SBATCH -J madm-cot-base
#SBATCH -o logs/run_cot_base.%j.out
#SBATCH -e logs/run_cot_base.%j.err
#SBATCH -p mi2104x
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 04:00:00

cd "${WORK}/madm"
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export HF_HOME=/work1/harangju/harang/.cache/huggingface

MODEL="gpt-5-nano-2025-08-07"
N=150

python scripts/run_MovieLens.py        --model "$MODEL" --n $N
python scripts/run_FEVEROUS.py         --model "$MODEL" --n $N
python scripts/run_LendingClub.py      --model "$MODEL" --n $N
python scripts/run_HotelBookings.py    --model "$MODEL" --n $N
python scripts/run_JFLEG.py            --model "$MODEL" --n $N
python scripts/run_MoralMachine.py     --model "$MODEL" --n $N
python scripts/run_Uber.py             --model "$MODEL" --n $N
python scripts/run_WikipediaToxicity.py --model "$MODEL" --n $N
python scripts/run_aime.py             --model "$MODEL" --n $N
