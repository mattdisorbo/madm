"""
GRPO Data Preparation: HotelBookings with hint conditions and cost ratios.

Generates frozen Call 1 predictions via Together API, then creates
training examples for each scenario × cost ratio combination.

Output: data/grpo_hotel_train (12,000) and data/grpo_hotel_holdout (3,000)

Usage:
    python scripts/grpo_prepare_hotel.py
"""

import os, re, random, json
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import openai

# Import dataset helpers from study3
from study3 import (
    load_hotel, hotel_prompt, hotel_conditions,
    HOTEL, parse_prediction,
)

MODEL = "Qwen/Qwen3.5-9B"
N_TRAIN_PER_CONDITION = int(os.environ.get("N_TRAIN", "200"))
N_HOLDOUT_PER_CONDITION = int(os.environ.get("N_HOLDOUT", "50"))
WORKERS = int(os.environ.get("WORKERS", "20"))
COST_RATIOS = [2, 4, 8, 10, 20, 50]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "../data")
TRAIN_DIR = os.path.join(DATA_DIR, "grpo_hotel_train")
HOLDOUT_DIR = os.path.join(DATA_DIR, "grpo_hotel_holdout")

client = openai.OpenAI(
    api_key=os.environ["TOGETHER_API_KEY"],
    base_url="https://api.together.xyz/v1",
)

predict_prompt = HOTEL["predict_prompt"]
escalate_prompt = HOTEL["escalate_prompt"]


def llm(messages, max_tokens=256):
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    r = client.chat.completions.create(
        model=MODEL, messages=messages, max_tokens=max_tokens,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    text = r.choices[0].message.content.strip()
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    return text


def make_cost_framing(ratio):
    return (
        f"The labor cost of escalating to a human is c_l = 1. "
        f"The cost of implementing a wrong answer is c_w = {ratio}. "
        f"A wrong implementation costs {ratio}x more than escalation."
    )


def process_sample(scenario, gt, hint, cond_name, base_rate):
    """Run frozen Call 1 for one scenario. Returns dict with prediction or None."""
    try:
        prompt = f"{scenario}\n\nHINT: {hint}\n\n{predict_prompt}"
        response = llm(prompt)
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


def expand_cost_ratios(sample_results):
    """Expand each sample into multiple training examples, one per cost ratio."""
    examples = []
    for s in sample_results:
        for ratio in COST_RATIOS:
            cost_framing = make_cost_framing(ratio)
            prompt_messages = [
                {"role": "user", "content": s["predict_prompt"]},
                {"role": "assistant", "content": s["prediction_text"]},
                {"role": "user", "content": cost_framing + "\n\n" + escalate_prompt},
            ]
            examples.append({
                "prompt": prompt_messages,
                "ground_truth": str(s["ground_truth"]),
                "prediction": str(s["prediction"]),
                "cost_ratio": str(ratio),
                "hint_condition": s["condition"],
                "base_rate": s["base_rate"],
            })
    return examples


if __name__ == "__main__":
    n_per_condition = N_TRAIN_PER_CONDITION + N_HOLDOUT_PER_CONDITION
    print(f"Model: {MODEL}")
    print(f"Per condition: {N_TRAIN_PER_CONDITION} train + {N_HOLDOUT_PER_CONDITION} holdout")
    print(f"Cost ratios: {COST_RATIOS}")
    print(f"Workers: {WORKERS}")

    print("Loading HotelBookings data...")
    df = load_hotel()
    conditions = hotel_conditions(df)
    gt_col = HOTEL["gt_col"]

    all_train = []
    all_holdout = []

    for cond in conditions:
        name = cond["name"]
        mask = cond["mask"]
        hint = cond["hint"]
        base_rate = cond["base_rate"]

        subset = df[mask]
        sample = subset.sample(n=min(n_per_condition, len(subset)), random_state=42)
        scenarios = [hotel_prompt(r) for _, r in sample.iterrows()]
        gts = [int(r[gt_col]) for _, r in sample.iterrows()]

        print(f"\n{'='*60}")
        print(f"  {name} (base_rate={base_rate:.0%}, n={len(sample)})")
        print(f"{'='*60}")

        # Run frozen Call 1 predictions via Together API
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
        n_train = min(N_TRAIN_PER_CONDITION, len(results))
        n_holdout = min(N_HOLDOUT_PER_CONDITION, len(results) - n_train)
        train_samples = results[:n_train]
        holdout_samples = results[n_train:n_train + n_holdout]

        # Expand by cost ratios
        train_examples = expand_cost_ratios(train_samples)
        holdout_examples = expand_cost_ratios(holdout_samples)

        all_train.extend(train_examples)
        all_holdout.extend(holdout_examples)

        pred_acc = np.mean([s["prediction"] == s["ground_truth"] for s in results])
        print(f"  Prediction accuracy: {pred_acc:.3f}")

    # Save as HF Datasets
    from datasets import Dataset

    print(f"\nTotal train examples: {len(all_train)}")
    print(f"Total holdout examples: {len(all_holdout)}")

    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(HOLDOUT_DIR, exist_ok=True)

    ds_train = Dataset.from_list(all_train)
    ds_train.save_to_disk(TRAIN_DIR)
    print(f"Saved train dataset to {TRAIN_DIR}")

    ds_holdout = Dataset.from_list(all_holdout)
    ds_holdout.save_to_disk(HOLDOUT_DIR)
    print(f"Saved holdout dataset to {HOLDOUT_DIR}")

    # Spot-check: print 3 examples across different cost ratios
    print(f"\n{'='*60}")
    print("SPOT CHECK: 3 example prompts")
    print(f"{'='*60}")
    for i, ex in enumerate(all_train[:3]):
        print(f"\n--- Example {i+1} (condition={ex['hint_condition']}, cost_ratio={ex['cost_ratio']}) ---")
        print(f"  GT={ex['ground_truth']}, Pred={ex['prediction']}")
        user_msg = ex["prompt"][-1]["content"]
        print(f"  Call 2 prompt (first 200 chars): {user_msg[:200]}...")

    # Summary by condition
    df_summary = pd.DataFrame(all_train)
    print(f"\n{'='*60}")
    print("TRAIN EXAMPLES BY CONDITION")
    print(f"{'='*60}")
    print(df_summary.groupby("hint_condition").size().to_string())
