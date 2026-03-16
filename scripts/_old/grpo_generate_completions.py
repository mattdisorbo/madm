"""
Generate multiple completions per prompt via Together API for offline REINFORCE.

For each training prompt, generates K completions and scores them with
the cost-based reward function. Saves (prompt, completion, reward, log_prob)
tuples for training.

Usage:
    python scripts/grpo_generate_completions.py
"""

import os, re, random
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from datasets import load_from_disk, Dataset
import openai

MODEL = "Qwen/Qwen2.5-7B-Instruct-Turbo"
K = 4  # completions per prompt
WORKERS = int(os.environ.get("WORKERS", "20"))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(BASE_DIR, "../data/grpo_hotel_train")
OUTPUT_PATH = os.path.join(BASE_DIR, "../data/grpo_hotel_completions")

client = openai.OpenAI(
    api_key=os.environ["TOGETHER_API_KEY"],
    base_url="https://api.together.xyz/v1",
)


def parse_decision(text):
    match = re.search(r'[01]', text.strip())
    if match:
        return int(match.group())
    low = text.lower()
    if 'implement' in low:
        return 0
    if 'escalat' in low:
        return 1
    return None


def compute_reward(escalate, pred_correct, R):
    if escalate is None:
        return -R
    if pred_correct and escalate == 0:
        return 0.0      # TN
    if not pred_correct and escalate == 1:
        return 0.0      # TP
    if pred_correct and escalate == 1:
        return -1.0     # FP
    if not pred_correct and escalate == 0:
        return -R       # FN
    return 0.0


def generate_completions(example, idx):
    """Generate K completions for one prompt."""
    try:
        r = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": example["prompt"]}],
            max_tokens=8,
            n=K,
            temperature=1.0,
            # No thinking mode for Qwen2.5
        )
    except Exception as e:
        return []

    pred_correct = str(example["prediction"]) == str(example["ground_truth"])
    R = float(example["cost_ratio"])

    results = []
    for choice in r.choices:
        text = choice.message.content.strip()
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
        escalate = parse_decision(text)
        reward = compute_reward(escalate, pred_correct, R)

        results.append({
            "prompt": example["prompt"],
            "completion": text,
            "reward": reward,
            "ground_truth": example["ground_truth"],
            "prediction": example["prediction"],
            "cost_ratio": example["cost_ratio"],
            "hint_condition": example["hint_condition"],
            "base_rate": example["base_rate"],
        })

    return results


if __name__ == "__main__":
    print(f"Loading dataset from {TRAIN_PATH}...", flush=True)
    dataset = load_from_disk(TRAIN_PATH)
    print(f"Dataset size: {len(dataset)}, generating {K} completions each", flush=True)

    all_results = []
    failed = 0

    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = {
            executor.submit(generate_completions, dataset[i], i): i
            for i in range(len(dataset))
        }
        for f in tqdm(as_completed(futures), total=len(futures), desc="Generating"):
            results = f.result()
            if results:
                all_results.extend(results)
            else:
                failed += 1

    print(f"\nDone: {len(all_results)} completions, {failed} failed prompts", flush=True)

    # Save
    ds = Dataset.from_list(all_results)
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    ds.save_to_disk(OUTPUT_PATH)
    print(f"Saved to {OUTPUT_PATH}", flush=True)

    # Stats
    rewards = [r["reward"] for r in all_results]
    print(f"Reward stats: mean={np.mean(rewards):.3f} std={np.std(rewards):.3f}")
    print(f"Reward distribution:")
    for v in sorted(set(rewards)):
        n = sum(1 for r in rewards if r == v)
        print(f"  {v:>6.1f}: {n} ({n/len(rewards):.1%})")
