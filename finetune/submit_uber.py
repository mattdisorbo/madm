import json
import openai
from pathlib import Path

client = openai.OpenAI()

HERE      = Path(__file__).parent
DATA_DIR  = HERE / "data"
JOBS_PATH = HERE / "jobs.json"
FT_MODEL  = "gpt-4o-mini-2024-07-18"

jsonl_path = DATA_DIR / "Uber.jsonl"
print(f"Uploading {jsonl_path} ...")
with open(jsonl_path, "rb") as f:
    file_resp = client.files.create(file=f, purpose="fine-tune")
file_id = file_resp.id
print(f"File uploaded: {file_id}")

print(f"Submitting fine-tuning job (model={FT_MODEL}) ...")
job = client.fine_tuning.jobs.create(
    training_file=file_id,
    model=FT_MODEL,
    suffix="uber",
)
print(f"Job submitted: {job.id}  status={job.status}")

jobs = {}
if JOBS_PATH.exists():
    with open(JOBS_PATH) as f:
        jobs = json.load(f)

jobs["Uber"] = {
    "job_id":  job.id,
    "file_id": file_id,
    "status":  job.status,
    "model":   FT_MODEL,
}
with open(JOBS_PATH, "w") as f:
    json.dump(jobs, f, indent=2)
print(f"Saved to {JOBS_PATH}")
