#!/bin/bash
#SBATCH -J dpo-perturb
#SBATCH -o logs/dpo_perturb.%j.out
#SBATCH -e logs/dpo_perturb.%j.err
#SBATCH -p mi2508x
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 06:00:00

cd "${WORK}/madm"
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export HF_HOME=/work1/harangju/harang/.cache/huggingface
export TOGETHER_API_KEY=unused

python scripts/dpo_eval_perturbations.py --adapter outputs/dpo_full
echo DONE
