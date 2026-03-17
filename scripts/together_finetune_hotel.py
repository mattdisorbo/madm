"""
Fine-tune a model on Together AI for cost-sensitive escalation.

Uses oracle labels based on condition-level optimal policy:
  escalate if P(wrong | condition) > 1/(1+R)

Usage:
  python scripts/together_finetune_hotel.py prepare   # create JSONL
  python scripts/together_finetune_hotel.py upload     # upload to Together
  python scripts/together_finetune_hotel.py train      # start fine-tuning
  python scripts/together_finetune_hotel.py status     # check status
  python scripts/together_finetune_hotel.py evaluate   # evaluate fine-tuned model
"""

import json
import os
import sys
import time
from pathlib import Path

from datasets import load_from_disk
from together import Together

DATA_DIR = Path("data")
OUTPUT_DIR = Path("data/together_hotel")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_JSONL = OUTPUT_DIR / "train.jsonl"
HOLDOUT_JSONL = OUTPUT_DIR / "holdout.jsonl"

# Together fine-tuning model
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct-Turbo"
TRAIN_MODEL = "Qwen/Qwen2.5-7B-Instruct"  # base model for fine-tuning

COST_RATIOS = [2, 4, 8, 10, 20, 50]


def compute_optimal_labels(df):
    """Compute condition-level optimal escalation labels.

    For each (hint_condition, cost_ratio) pair:
      P(wrong) = 1 - empirical_accuracy(condition)
      escalate if P(wrong) > 1/(1+R)
    """
    df = df.copy()
    df["correct"] = df["prediction"] == df["ground_truth"]
    cond_acc = df.groupby("hint_condition")["correct"].mean().to_dict()

    labels = []
    for _, row in df.iterrows():
        acc = cond_acc[row["hint_condition"]]
        pw = 1 - acc
        R = int(row["cost_ratio"])
        threshold = 1 / (1 + R)
        labels.append("1" if pw > threshold else "0")

    df["optimal_label"] = labels
    return df


def to_together_sft(df, output_path):
    """Convert to Together SFT JSONL format (chat messages)."""
    with open(output_path, "w") as f:
        for _, row in df.iterrows():
            entry = {
                "messages": [
                    {"role": "user", "content": row["prompt"]},
                    {"role": "assistant", "content": row["optimal_label"]},
                ]
            }
            f.write(json.dumps(entry) + "\n")
    print(f"Wrote {len(df)} examples to {output_path}")


def cmd_prepare():
    """Create JSONL files with optimal labels."""
    print("Loading training data...")
    train_df = load_from_disk(str(DATA_DIR / "grpo_hotel_train")).to_pandas()
    holdout_df = load_from_disk(str(DATA_DIR / "grpo_hotel_holdout")).to_pandas()

    print("\nComputing optimal labels for training set...")
    train_df = compute_optimal_labels(train_df)

    print("\nComputing optimal labels for holdout set...")
    holdout_df = compute_optimal_labels(holdout_df)

    # Print label distribution
    print("\n=== Training set ===")
    print(f"Total: {len(train_df)}")
    print(f"Escalate: {(train_df['optimal_label'] == '1').sum()}")
    print(f"Implement: {(train_df['optimal_label'] == '0').sum()}")

    print("\nLabels by (condition, cost_ratio):")
    for hc in sorted(train_df["hint_condition"].unique()):
        sub = train_df[train_df["hint_condition"] == hc]
        acc = (sub["prediction"] == sub["ground_truth"]).mean()
        labels_by_r = {}
        for r in COST_RATIOS:
            r_sub = sub[sub["cost_ratio"] == str(r)]
            if len(r_sub) > 0:
                labels_by_r[r] = r_sub["optimal_label"].iloc[0]
        label_str = "  ".join(f"R={r}:{labels_by_r.get(r, '?')}" for r in COST_RATIOS)
        print(f"  {hc:<25} acc={acc:.3f}  {label_str}")

    print("\n=== Holdout set ===")
    print(f"Total: {len(holdout_df)}")
    print(f"Escalate: {(holdout_df['optimal_label'] == '1').sum()}")
    print(f"Implement: {(holdout_df['optimal_label'] == '0').sum()}")

    # Write JSONL
    to_together_sft(train_df, TRAIN_JSONL)
    to_together_sft(holdout_df, HOLDOUT_JSONL)

    print(f"\nFiles written to {OUTPUT_DIR}/")


def cmd_upload():
    """Upload JSONL to Together."""
    client = Together()

    print(f"Uploading {TRAIN_JSONL}...")
    train_resp = client.files.upload(file=str(TRAIN_JSONL), purpose="fine-tune")
    print(f"  Train file ID: {train_resp.id}")

    print(f"Uploading {HOLDOUT_JSONL}...")
    holdout_resp = client.files.upload(file=str(HOLDOUT_JSONL), purpose="fine-tune")
    print(f"  Holdout file ID: {holdout_resp.id}")

    # Save file IDs
    ids_path = OUTPUT_DIR / "file_ids.json"
    with open(ids_path, "w") as f:
        json.dump({"train": train_resp.id, "holdout": holdout_resp.id}, f, indent=2)
    print(f"\nFile IDs saved to {ids_path}")


def cmd_train():
    """Start fine-tuning on Together."""
    client = Together()

    ids_path = OUTPUT_DIR / "file_ids.json"
    with open(ids_path) as f:
        file_ids = json.load(f)

    print(f"Starting fine-tuning on {TRAIN_MODEL}...")
    print(f"  Train file: {file_ids['train']}")
    print(f"  Holdout file: {file_ids['holdout']}")

    ft_resp = client.fine_tuning.create(
        training_file=file_ids["train"],
        validation_file=file_ids["holdout"],
        model=TRAIN_MODEL,
        n_epochs=3,
        n_checkpoints=1,
        lora=True,
        learning_rate=1e-5,
        batch_size="max",
        suffix="hotel-escalation",
        train_on_inputs=False,
    )

    print(f"\nFine-tuning job created!")
    print(f"  Job ID: {ft_resp.id}")

    # Save job ID
    job_path = OUTPUT_DIR / "job_id.txt"
    with open(job_path, "w") as f:
        f.write(ft_resp.id)
    print(f"  Job ID saved to {job_path}")


def cmd_status():
    """Check fine-tuning job status."""
    client = Together()

    job_path = OUTPUT_DIR / "job_id.txt"
    job_id = job_path.read_text().strip()

    status = client.fine_tuning.retrieve(job_id)
    print(f"Job ID: {job_id}")
    print(f"Status: {status.status}")
    print(f"Model: {status.model}")
    if hasattr(status, "model_output_name") and status.model_output_name:
        print(f"Output model: {status.model_output_name}")
        model_path = OUTPUT_DIR / "model_name.txt"
        with open(model_path, "w") as f:
            f.write(status.model_output_name)
        print(f"Model name saved to {model_path}")
    if hasattr(status, "events") and status.events:
        print("\nRecent events:")
        for event in status.events[-5:]:
            print(f"  {event}")


def cmd_evaluate():
    """Evaluate fine-tuned model vs baseline on holdout set.

    Usage:
      evaluate                 # run both baseline and finetuned
      evaluate --model NAME    # run specific model with label "custom"
      evaluate --baseline-only # run baseline only
      evaluate --ft-only       # run finetuned only
    """
    client = Together()

    holdout_df = load_from_disk(str(DATA_DIR / "grpo_hotel_holdout")).to_pandas()
    holdout_df = compute_optimal_labels(holdout_df)

    # Determine which models to evaluate
    models_to_eval = []
    if "--baseline-only" in sys.argv:
        models_to_eval = [(BASE_MODEL, "baseline")]
    elif "--ft-only" in sys.argv or "--model" in sys.argv:
        if "--model" in sys.argv:
            idx = sys.argv.index("--model")
            ft_model = sys.argv[idx + 1]
            label = sys.argv[idx + 2] if len(sys.argv) > idx + 2 and not sys.argv[idx + 2].startswith("--") else "finetuned"
        else:
            for p in [OUTPUT_DIR / "model_name_maverick.txt", OUTPUT_DIR / "model_name.txt"]:
                if p.exists():
                    ft_model = p.read_text().strip()
                    break
            else:
                print("No fine-tuned model found.")
                return
            label = "finetuned"
        models_to_eval = [(ft_model, label)]
    else:
        ft_model = None
        for p in [OUTPUT_DIR / "model_name_maverick.txt", OUTPUT_DIR / "model_name.txt"]:
            if p.exists():
                ft_model = p.read_text().strip()
                break
        if not ft_model:
            print("No fine-tuned model found. Running baseline only.")
            models_to_eval = [(BASE_MODEL, "baseline")]
        else:
            models_to_eval = [(BASE_MODEL, "baseline"), (ft_model, "finetuned")]

    results = []
    for model_name, label in models_to_eval:
        print(f"\n=== Evaluating {label} ({model_name}) ===")

        predictions = []
        for i, (_, row) in enumerate(holdout_df.iterrows()):
            if i % 100 == 0:
                print(f"  {i}/{len(holdout_df)}...", flush=True)

            for attempt in range(5):
                try:
                    resp = client.chat.completions.create(
                        model=model_name,
                        messages=[{"role": "user", "content": row["prompt"]}],
                        max_tokens=1,
                        temperature=0,
                    )
                    pred = resp.choices[0].message.content.strip()
                    predictions.append(pred)
                    break
                except Exception as e:
                    if attempt < 4:
                        time.sleep(2 ** attempt)
                    else:
                        raise

        holdout_df[f"pred_{label}"] = predictions

        # Compute metrics per (condition, cost_ratio)
        print(f"\n  {'Condition':<25} {'R':>3} {'Esc%':>5} {'Opt%':>5} {'Match':>5}")
        print("  " + "-" * 50)

        for hc in sorted(holdout_df["hint_condition"].unique()):
            for r in COST_RATIOS:
                sub = holdout_df[
                    (holdout_df["hint_condition"] == hc)
                    & (holdout_df["cost_ratio"] == str(r))
                ]
                if len(sub) == 0:
                    continue
                esc_rate = (sub[f"pred_{label}"] == "1").mean()
                opt_rate = (sub["optimal_label"] == "1").mean()
                match = (sub[f"pred_{label}"] == sub["optimal_label"]).mean()
                results.append({
                    "model": label,
                    "condition": hc,
                    "cost_ratio": r,
                    "esc_rate": esc_rate,
                    "opt_rate": opt_rate,
                    "match": match,
                    "n": len(sub),
                })
                print(f"  {hc:<25} {r:>3} {esc_rate:>5.1%} {opt_rate:>5.1%} {match:>5.1%}")

    # Save results
    import pandas as pd
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / "eval_results.csv", index=False)
    print(f"\nResults saved to {OUTPUT_DIR / 'eval_results.csv'}")

    # Summary
    print("\n=== SUMMARY ===")
    for label in ["baseline", "finetuned"]:
        sub = results_df[results_df["model"] == label]
        print(f"\n{label}:")
        print(f"  Overall match with optimal: {sub['match'].mean():.1%}")
        print(f"  Mean escalation rate: {sub['esc_rate'].mean():.1%}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "prepare":
        cmd_prepare()
    elif cmd == "upload":
        cmd_upload()
    elif cmd == "train":
        cmd_train()
    elif cmd == "status":
        cmd_status()
    elif cmd == "evaluate":
        cmd_evaluate()
    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)
        sys.exit(1)
