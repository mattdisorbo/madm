#!/bin/bash
#SBATCH -J madm-together
#SBATCH -o logs/fast_together_%j.out
#SBATCH -e logs/fast_together_%j.err
#SBATCH -p mi2101x
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00

cd "${WORK}/madm"
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export TOGETHER_API_KEY=tgp_v1_-VyuWxKs2Uxs0Q5-0VZLCvFjl-e7rWF3V5jDXNO1rT4

uv pip install openai

echo "--- Starting Together API runs (cluster datasets) ---"
python scripts/run_together.py \
    --datasets MoralMachine FEVEROUS WikipediaToxicity MovieLens \
    --n 100 --workers 20
echo "--- Done cluster datasets ---"
