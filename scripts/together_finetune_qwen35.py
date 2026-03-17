"""
Fine-tune Qwen3.5-9B for cost-sensitive escalation on HotelBookings.

Uses oracle labels derived from the hint's base rate:
  escalate if (1 - base_rate) > 1/(1+R)

Steps:
  python scripts/together_finetune_qwen35.py prepare   # generate predictions + JSONL
  python scripts/together_finetune_qwen35.py upload     # upload to Together
  python scripts/together_finetune_qwen35.py train      # start fine-tuning
  python scripts/together_finetune_qwen35.py status     # check status
  python scripts/together_finetune_qwen35.py evaluate   # evaluate fine-tuned model
"""

import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import openai
import pandas as pd
from tqdm import tqdm

# Import dataset helpers from study3
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from study3 import (
    load_hotel, hotel_prompt, hotel_conditions,
    HOTEL, parse_prediction,
)

MODEL = "Qwen/Qwen3.5-9B"
TRAIN_MODEL = "Qwen/Qwen3-8B"  # closest fine-tunable model to Qwen3.5-9B
N_TRAIN_PER_HINT = 200
N_HOLDOUT_PER_HINT = 50
WORKERS = int(os.environ.get("WORKERS", "20"))
COST_RATIOS = [2, 4, 8, 10, 20, 50]

OUTPUT_DIR = Path("results/fine-tuning-qwen3.5-9b")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_JSONL = OUTPUT_DIR / "train.jsonl"
HOLDOUT_JSONL = OUTPUT_DIR / "holdout.jsonl"

predict_prompt = HOTEL["predict_prompt"]
escalate_prompt = HOTEL["escalate_prompt"]


def get_client():
    return openai.OpenAI(
        api_key=os.environ["TOGETHER_API_KEY"],
        base_url="https://api.together.xyz/v1",
    )


def get_together_client():
    from together import Together
    return Together()


def llm_predict(client, prompt, max_tokens=256):
    """Run a single LLM prediction."""
    r = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    text = r.choices[0].message.content.strip()
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    return text


def oracle_label(base_rate, cost_ratio):
    """Compute oracle escalation label from hint base rate and cost ratio."""
    pw = 1 - base_rate
    threshold = 1 / (1 + cost_ratio)
    return "1" if pw > threshold else "0"


def make_cost_framing(ratio):
    return (
        f"Implementing a wrong answer costs {ratio}x more than escalating.\n\n"
        f"{escalate_prompt}"
    )


def process_sample(scenario, gt, hint, cond_name, base_rate):
    """Run Call 1 prediction for one scenario."""
    client = get_client()
    try:
        prompt = f"{scenario}\n\n{hint}\n\n{predict_prompt}"
        response = llm_predict(client, prompt)
        pred = parse_prediction(response)
        if pred is None:
            return None
        return {
            "scenario": scenario,
            "hint": hint,
            "condition": cond_name,
            "base_rate": base_rate,
            "ground_truth": gt,
            "prediction": pred,
            "prediction_text": response,
            "predict_prompt": prompt,
        }
    except Exception as e:
        print(f"  Error: {e}")
        return None


def expand_to_sft(sample_results, output_path):
    """Expand samples across cost ratios and write Together SFT JSONL."""
    with open(output_path, "w") as f:
        count = 0
        for s in sample_results:
            for ratio in COST_RATIOS:
                label = oracle_label(s["base_rate"], ratio)
                entry = {
                    "messages": [
                        {"role": "user", "content": s["predict_prompt"]},
                        {"role": "assistant", "content": s["prediction_text"]},
                        {"role": "user", "content": make_cost_framing(ratio)},
                        {"role": "assistant", "content": label},
                    ]
                }
                f.write(json.dumps(entry) + "\n")
                count += 1
    print(f"Wrote {count} examples to {output_path}")
    return count


def cmd_prepare():
    """Generate predictions and create JSONL files."""
    print(f"Model: {MODEL}")
    print(f"Per hint: {N_TRAIN_PER_HINT} train + {N_HOLDOUT_PER_HINT} holdout")
    print(f"Cost ratios: {COST_RATIOS}")
    print(f"Workers: {WORKERS}")

    print("Loading HotelBookings data...")
    df = load_hotel()
    conditions = hotel_conditions(df)
    gt_col = HOTEL["gt_col"]

    all_train = []
    all_holdout = []
    n_per_hint = N_TRAIN_PER_HINT + N_HOLDOUT_PER_HINT

    for cond in conditions:
        name = cond["name"]
        mask = cond["mask"]
        hint = cond["hint"]
        base_rate = cond["base_rate"]

        subset = df[mask]
        sample = subset.sample(n=min(n_per_hint, len(subset)), random_state=42)
        scenarios = [hotel_prompt(r) for _, r in sample.iterrows()]
        gts = [int(r[gt_col]) for _, r in sample.iterrows()]

        print(f"\n{'='*60}")
        print(f"  {name} (base_rate={base_rate:.0%}, n={len(sample)})")
        print(f"{'='*60}")

        # Run Call 1 predictions
        import random
        results = []
        failed = 0
        with ThreadPoolExecutor(max_workers=WORKERS) as executor:
            futures = {
                executor.submit(process_sample, s, g, hint, name, base_rate): i
                for i, (s, g) in enumerate(zip(scenarios, gts))
            }
            for f in tqdm(as_completed(futures), total=len(futures), desc=name):
                result = f.result()
                if result:
                    results.append(result)
                else:
                    failed += 1

        print(f"  {len(results)} ok, {failed} failed")

        # Split into train/holdout
        random.seed(42)
        random.shuffle(results)
        n_train = min(N_TRAIN_PER_HINT, len(results))
        n_holdout = min(N_HOLDOUT_PER_HINT, len(results) - n_train)
        all_train.extend(results[:n_train])
        all_holdout.extend(results[n_train:n_train + n_holdout])

        pred_acc = sum(s["prediction"] == s["ground_truth"] for s in results) / len(results)
        print(f"  Prediction accuracy: {pred_acc:.3f}")

    # Write JSONL
    print(f"\n{'='*60}")
    expand_to_sft(all_train, TRAIN_JSONL)
    expand_to_sft(all_holdout, HOLDOUT_JSONL)

    # Summary
    print(f"\nOracle labels by (hint, cost_ratio):")
    for cond in conditions:
        labels = {r: oracle_label(cond["base_rate"], r) for r in COST_RATIOS}
        label_str = "  ".join(f"R={r}:{labels[r]}" for r in COST_RATIOS)
        print(f"  {cond['name']:<25} base_rate={cond['base_rate']:.3f}  {label_str}")

    print(f"\nFiles written to {OUTPUT_DIR}/")


def cmd_upload():
    """Upload JSONL to Together."""
    client = get_together_client()

    print(f"Uploading {TRAIN_JSONL}...")
    train_resp = client.files.upload(file=str(TRAIN_JSONL), purpose="fine-tune")
    print(f"  Train file ID: {train_resp.id}")

    print(f"Uploading {HOLDOUT_JSONL}...")
    holdout_resp = client.files.upload(file=str(HOLDOUT_JSONL), purpose="fine-tune")
    print(f"  Holdout file ID: {holdout_resp.id}")

    ids_path = OUTPUT_DIR / "file_ids.json"
    with open(ids_path, "w") as f:
        json.dump({"train": train_resp.id, "holdout": holdout_resp.id}, f, indent=2)
    print(f"\nFile IDs saved to {ids_path}")


def cmd_train():
    """Start fine-tuning on Together."""
    client = get_together_client()

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

    job_path = OUTPUT_DIR / "job_id.txt"
    with open(job_path, "w") as f:
        f.write(ft_resp.id)
    print(f"  Job ID saved to {job_path}")


def cmd_status():
    """Check fine-tuning job status."""
    client = get_together_client()

    job_path = OUTPUT_DIR / "job_id.txt"
    job_id = job_path.read_text().strip()

    status = client.fine_tuning.retrieve(job_id)
    print(f"Job ID: {job_id}")
    print(f"Status: {status.status}")
    print(f"Model: {status.model}")
    if hasattr(status, "total_price") and status.total_price:
        print(f"Cost: ${status.total_price / 1e9:.2f}")
    if hasattr(status, "x_model_output_name") and status.x_model_output_name:
        print(f"Output model: {status.x_model_output_name}")
        model_path = OUTPUT_DIR / "model_name.txt"
        with open(model_path, "w") as f:
            f.write(status.x_model_output_name)
        print(f"Model name saved to {model_path}")
    if hasattr(status, "events") and status.events:
        print("\nRecent events:")
        for event in status.events[-5:]:
            print(f"  [{event.type}] {event.message}")


def cmd_evaluate():
    """Evaluate fine-tuned model vs baseline on holdout set."""
    client = get_together_client()

    # Load holdout JSONL
    holdout_rows = []
    with open(HOLDOUT_JSONL) as f:
        for line in f:
            holdout_rows.append(json.loads(line))

    # Determine which models to evaluate
    models_to_eval = []
    if "--baseline-only" in sys.argv:
        models_to_eval = [(MODEL, "baseline")]
    elif "--ft-only" in sys.argv or "--model" in sys.argv:
        if "--model" in sys.argv:
            idx = sys.argv.index("--model")
            ft_model = sys.argv[idx + 1]
        else:
            model_path = OUTPUT_DIR / "model_name.txt"
            if not model_path.exists():
                print("No fine-tuned model found.")
                return
            ft_model = model_path.read_text().strip()
        models_to_eval = [(ft_model, "finetuned")]
    else:
        model_path = OUTPUT_DIR / "model_name.txt"
        if model_path.exists():
            ft_model = model_path.read_text().strip()
            models_to_eval = [(MODEL, "baseline"), (ft_model, "finetuned")]
        else:
            print("No fine-tuned model found. Running baseline only.")
            models_to_eval = [(MODEL, "baseline")]

    # Parse hint condition and cost ratio from each holdout row
    holdout_meta = []
    for row in holdout_rows:
        prompt = row["messages"][2]["content"]  # cost framing + escalate prompt
        m = re.search(r'(\d+)x more than escalating', prompt)
        cost_ratio = int(m.group(1)) if m else 0

        # Get hint condition from the user prompt
        user_prompt = row["messages"][0]["content"]
        # Find which condition this matches by checking hints
        df_hotel = load_hotel()
        conds = hotel_conditions(df_hotel)
        base_rate = 0
        hint_name = "unknown"
        for cond in conds:
            if cond["hint"] in user_prompt:
                base_rate = cond["base_rate"]
                hint_name = cond["name"]
                break

        optimal = oracle_label(base_rate, cost_ratio)
        holdout_meta.append({
            "hint_condition": hint_name,
            "cost_ratio": cost_ratio,
            "base_rate": base_rate,
            "optimal_label": optimal,
            "messages": row["messages"][:3],  # user/assistant/user (no oracle answer)
        })

    results = []
    all_predictions = []

    for model_name, label in models_to_eval:
        print(f"\n=== Evaluating {label} ({model_name}) ===")

        predictions = []
        for i, meta in enumerate(holdout_meta):
            if i % 100 == 0:
                print(f"  {i}/{len(holdout_meta)}...", flush=True)

            for attempt in range(5):
                try:
                    resp = client.chat.completions.create(
                        model=model_name,
                        messages=meta["messages"],
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

        # Compute metrics per (hint, cost_ratio)
        print(f"\n  {'Hint':<25} {'R':>3} {'Esc%':>5} {'Opt%':>5} {'Match':>5}")
        print("  " + "-" * 50)

        for cond in hotel_conditions(load_hotel()):
            for r in COST_RATIOS:
                indices = [j for j, m in enumerate(holdout_meta)
                           if m["hint_condition"] == cond["name"] and m["cost_ratio"] == r]
                if not indices:
                    continue
                preds = [predictions[j] for j in indices]
                opts = [holdout_meta[j]["optimal_label"] for j in indices]
                esc_rate = sum(p == "1" for p in preds) / len(preds)
                opt_rate = sum(o == "1" for o in opts) / len(opts)
                match = sum(p == o for p, o in zip(preds, opts)) / len(preds)
                results.append({
                    "model": label,
                    "hint": cond["name"],
                    "cost_ratio": r,
                    "esc_rate": esc_rate,
                    "opt_rate": opt_rate,
                    "match": match,
                    "n": len(preds),
                })
                print(f"  {cond['name']:<25} {r:>3} {esc_rate:>5.1%} {opt_rate:>5.1%} {match:>5.1%}")

        # Save row-level predictions
        for j, pred in enumerate(predictions):
            all_predictions.append({
                "model": label,
                "hint_condition": holdout_meta[j]["hint_condition"],
                "cost_ratio": holdout_meta[j]["cost_ratio"],
                "base_rate": holdout_meta[j]["base_rate"],
                "optimal_label": holdout_meta[j]["optimal_label"],
                "prediction": pred,
            })

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / "eval_results.csv", index=False)
    print(f"\nResults saved to {OUTPUT_DIR / 'eval_results.csv'}")

    pred_df = pd.DataFrame(all_predictions)
    pred_df.to_csv(OUTPUT_DIR / "eval_predictions.csv", index=False)
    print(f"Row-level predictions saved to {OUTPUT_DIR / 'eval_predictions.csv'}")

    # Summary
    print("\n=== SUMMARY ===")
    for _, lbl in models_to_eval:
        sub = results_df[results_df["model"] == lbl]
        if len(sub) == 0:
            continue
        print(f"\n{lbl}:")
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
