#!/usr/bin/env python3
"""
Check the status of fine-tuning jobs and collect finished model IDs.

Reads finetune/jobs.json and prints current status for each job.
Saves completed model IDs to finetune/model_ids.json.

Usage:
    uv run python scripts/finetune/check_status.py

Once a model ID appears, pass it to any run_*.py script via:
    uv run python scripts/run_LendingClub.py --model <ft:gpt-4.1:...> --n 150
"""

import os, json
import openai

client = openai.OpenAI()

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
JOBS_PATH    = os.path.join(SCRIPT_DIR, "../../finetune/jobs.json")
MODELS_PATH  = os.path.join(SCRIPT_DIR, "../../finetune/model_ids.json")

with open(JOBS_PATH) as f:
    jobs = json.load(f)

# Load any previously saved model IDs
try:
    with open(MODELS_PATH) as f:
        model_ids = json.load(f)
except FileNotFoundError:
    model_ids = {}

print(f"{'Dataset':<22} {'Status':<20} {'Model ID'}")
print("-" * 80)

for dataset, info in jobs.items():
    job = client.fine_tuning.jobs.retrieve(info["job_id"])
    status   = job.status
    model_id = job.fine_tuned_model or ""
    print(f"{dataset:<22} {status:<20} {model_id}")
    if model_id:
        model_ids[dataset] = model_id

with open(MODELS_PATH, "w") as f:
    json.dump(model_ids, f, indent=2)

completed = [d for d in model_ids]
pending   = [d for d in jobs if d not in model_ids]

print(f"\nCompleted ({len(completed)}): {completed}")
print(f"Pending   ({len(pending)}):   {pending}")
if model_ids:
    print(f"\nModel IDs saved to {MODELS_PATH}")
