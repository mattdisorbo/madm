#!/usr/bin/env python3
"""
Upload JSONL files and submit fine-tuning jobs to OpenAI.

Run after prepare_data.py. Saves job IDs to finetune/jobs.json.

Usage:
    uv run python scripts/finetune/submit_jobs.py
"""

import os, json
import openai

client = openai.OpenAI()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(SCRIPT_DIR, "../../finetune/data")
JOBS_PATH  = os.path.join(SCRIPT_DIR, "../../finetune/jobs.json")

FT_MODEL = "gpt-4.1"

DATASETS = [
    "LendingClub",
    "HotelBookings",
    "MoralMachine",
    "Uber",
    "MovieLens",
    "AIME",
    "JFLEG",
    "FEVEROUS",
    "WikipediaToxicity",
]

jobs = {}
for dataset in DATASETS:
    jsonl_path = os.path.join(DATA_DIR, f"{dataset}.jsonl")
    if not os.path.exists(jsonl_path):
        print(f"[SKIP] {dataset}: {jsonl_path} not found — run prepare_data.py first")
        continue

    print(f"[{dataset}] Uploading {jsonl_path} ...")
    with open(jsonl_path, "rb") as f:
        file_resp = client.files.create(file=f, purpose="fine-tune")
    file_id = file_resp.id
    print(f"[{dataset}] File uploaded: {file_id}")

    print(f"[{dataset}] Submitting fine-tuning job (model={FT_MODEL}) ...")
    job = client.fine_tuning.jobs.create(
        training_file=file_id,
        model=FT_MODEL,
        suffix=dataset.lower().replace("wikipedia", "wiki"),
    )
    jobs[dataset] = {
        "job_id":   job.id,
        "file_id":  file_id,
        "status":   job.status,
        "model":    FT_MODEL,
    }
    print(f"[{dataset}] Job submitted: {job.id}  status={job.status}")

with open(JOBS_PATH, "w") as f:
    json.dump(jobs, f, indent=2)
print(f"\nSaved job info to {JOBS_PATH}")
print("Run check_status.py to monitor progress and retrieve model IDs.")
