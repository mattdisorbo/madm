"""
GRPO Evaluation: HotelBookings + out-of-distribution datasets.

Runs the 2-call pipeline (predict frozen, escalate with LoRA or baseline)
across multiple datasets and cost ratios. Produces per-condition metrics
and a side-by-side baseline vs GRPO dashboard.

Usage:
    python scripts/grpo_eval_hotel.py [MODEL] [--no-lora] [--lora-path PATH]
"""

import os, re, sys, random
import pandas as pd
import numpy as np
from tqdm import tqdm

# Import dataset registries from study3
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from study3 import (
    load_hotel, hotel_prompt, hotel_conditions, HOTEL,
    load_moral, moral_prompt, moral_conditions, MORAL,
    load_wiki, wiki_prompt, wiki_conditions, WIKI,
    load_lending, lending_prompt, lending_conditions, LENDING,
    parse_prediction, parse_escalation_cot,
)

MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3.5-9B"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LORA_PATH = os.path.join(BASE_DIR, "../outputs/grpo_hotel")
RESULTS_DIR = os.path.join(BASE_DIR, "../results/grpo_hotel")
N_PER_CONDITION = int(os.environ.get("N_PER_CONDITION", "50"))

# Cost ratios: training set + held-out (6 and 15 never seen in training)
COST_RATIOS = [2, 4, 6, 8, 10, 15, 20, 50]

EVAL_DATASETS = {
    "HotelBookings": HOTEL,
    "MoralMachine": MORAL,
    "WikipediaToxicity": WIKI,
    "LendingClub": LENDING,
}

for i, arg in enumerate(sys.argv):
    if arg == "--lora-path" and i + 1 < len(sys.argv):
        LORA_PATH = sys.argv[i + 1]
    if arg == "--n-per-condition" and i + 1 < len(sys.argv):
        N_PER_CONDITION = int(sys.argv[i + 1])

USE_LORA = "--no-lora" not in sys.argv
os.makedirs(RESULTS_DIR, exist_ok=True)


def make_cost_framing(ratio):
    return (
        f"The labor cost of escalating to a human is c_l = 1. "
        f"The cost of implementing a wrong answer is c_w = {ratio}. "
        f"A wrong implementation costs {ratio}x more than escalation."
    )


# --- Model setup ---
print(f"Loading base model {MODEL}...", flush=True)
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline as hf_pipeline
import torch

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="auto")

predict_pipe = hf_pipeline("text-generation", model=model, tokenizer=tokenizer)

if USE_LORA:
    print(f"Loading LoRA adapter from {LORA_PATH}...", flush=True)
    from peft import PeftModel
    escalate_model = PeftModel.from_pretrained(model, LORA_PATH)
    escalate_model.eval()
    escalate_pipe = hf_pipeline("text-generation", model=escalate_model, tokenizer=tokenizer)
    print("LoRA adapter loaded.", flush=True)
else:
    print("Running baseline evaluation (no LoRA).", flush=True)
    escalate_pipe = predict_pipe

label = "grpo" if USE_LORA else "baseline"
print(f"Mode: {label}", flush=True)


def run_pipe(pipe, messages, max_new_tokens=2048):
    """Run a pipeline with chat messages."""
    if "Qwen3.5" in MODEL:
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
    elif "Qwen3" in MODEL:
        messages = [{"role": "system", "content": "/no_think"}] + messages
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
    else:
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
    out = pipe(formatted, max_new_tokens=max_new_tokens, return_full_text=False)
    text = out[0]["generated_text"]
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    return text


def evaluate_dataset(ds_name, ds_config):
    """Evaluate one dataset across all conditions and cost ratios."""
    print(f"\n{'='*70}")
    print(f"EVALUATING: {ds_name} ({label})")
    print(f"{'='*70}")

    df = ds_config["load"]()
    conditions = ds_config["conditions"](df)
    gt_col = ds_config["gt_col"]
    ds_predict_prompt = ds_config["predict_prompt"]
    ds_escalate_prompt = ds_config["escalate_prompt"]
    create_prompt = ds_config["prompt"]

    all_results = []

    for cond in conditions:
        cond_name = cond["name"]
        mask = cond["mask"]
        hint = cond["hint"]
        base_rate = cond["base_rate"]

        subset = df[mask]
        sample = subset.sample(n=min(N_PER_CONDITION, len(subset)), random_state=42)

        print(f"\n  {cond_name} (base_rate={base_rate:.0%}, n={len(sample)})")

        for idx, row in tqdm(sample.iterrows(), total=len(sample),
                             desc=f"  {cond_name}", leave=False):
            scenario = create_prompt(row)
            gt = int(row[gt_col])

            # Call 1: Predict (frozen, no LoRA)
            try:
                prompt = f"{scenario}\n\nHINT: {hint}\n\n{ds_predict_prompt}"
                thought = run_pipe(predict_pipe,
                                   [{"role": "user", "content": prompt}])
                pred = parse_prediction(thought)
            except Exception as e:
                continue
            if pred is None:
                continue

            pred_correct = (pred == gt)

            # Call 2: Escalate for each cost ratio
            for ratio in COST_RATIOS:
                cost_framing = make_cost_framing(ratio)
                esc_prompt = cost_framing + "\n\n" + ds_escalate_prompt

                try:
                    esc_text = run_pipe(
                        escalate_pipe,
                        [
                            {"role": "user", "content": prompt},
                            {"role": "assistant", "content": thought},
                            {"role": "user", "content": esc_prompt},
                        ],
                        max_new_tokens=256,
                    )
                    escalate = parse_escalation_cot(esc_text)
                except Exception as e:
                    continue
                if escalate is None:
                    continue

                # Compute cost
                R = float(ratio)
                if pred_correct and escalate == 0:
                    cost = 0.0       # TN
                elif not pred_correct and escalate == 1:
                    cost = 0.0       # TP (human corrects)
                elif pred_correct and escalate == 1:
                    cost = 1.0       # FP (unnecessary escalation)
                else:
                    cost = R         # FN (wrong answer implemented)

                # Theoretical optimal threshold
                p_star = 1 - 1 / (1 + R)

                all_results.append({
                    "dataset": ds_name,
                    "condition": cond_name,
                    "base_rate": base_rate,
                    "cost_ratio": ratio,
                    "ground_truth": gt,
                    "prediction": pred,
                    "pred_correct": pred_correct,
                    "escalate": escalate,
                    "cost": cost,
                    "p_star": p_star,
                })

    return all_results


if __name__ == "__main__":
    all_results = []

    for ds_name, ds_config in EVAL_DATASETS.items():
        results = evaluate_dataset(ds_name, ds_config)
        all_results.extend(results)

    df = pd.DataFrame(all_results)

    # Save raw results
    raw_path = os.path.join(RESULTS_DIR, f"eval_{label}_all.csv")
    df.to_csv(raw_path, index=False)
    print(f"\nSaved raw results to {raw_path}")

    # Per-dataset CSV
    for ds_name in EVAL_DATASETS:
        ds_df = df[df["dataset"] == ds_name]
        ds_path = os.path.join(RESULTS_DIR, f"eval_{label}_{ds_name}.csv")
        ds_df.to_csv(ds_path, index=False)

    # --- Dashboard ---
    print(f"\n{'='*70}")
    print(f"DASHBOARD ({label})")
    print(f"{'='*70}")

    for ds_name in EVAL_DATASETS:
        ds_df = df[df["dataset"] == ds_name]
        if len(ds_df) == 0:
            continue

        print(f"\n--- {ds_name} ---")
        print(f"{'condition':<25} {'R':>3} {'esc_rate':>8} {'p*':>6} "
              f"{'gap':>6} {'avg_cost':>8} {'FP_rate':>7} {'FN_rate':>7} "
              f"{'esc_acc':>7} {'n':>5}")

        for ratio in COST_RATIOS:
            ratio_df = ds_df[ds_df["cost_ratio"] == ratio]
            if len(ratio_df) == 0:
                continue

            for cond_name in ratio_df["condition"].unique():
                cond_df = ratio_df[ratio_df["condition"] == cond_name]
                n = len(cond_df)
                esc_rate = cond_df["escalate"].mean()
                p_star = cond_df["p_star"].iloc[0]
                avg_cost = cond_df["cost"].mean()

                # FP/FN rates
                correct = cond_df[cond_df["pred_correct"]]
                wrong = cond_df[~cond_df["pred_correct"]]
                fp_rate = correct["escalate"].mean() if len(correct) > 0 else float('nan')
                fn_rate = (1 - wrong["escalate"].mean()) if len(wrong) > 0 else float('nan')

                # Escalation accuracy
                tp = ((cond_df["escalate"] == 1) & (~cond_df["pred_correct"])).sum()
                tn = ((cond_df["escalate"] == 0) & (cond_df["pred_correct"])).sum()
                esc_acc = (tp + tn) / n

                # Implied p* (escalation rate as threshold proxy)
                implied_gap = esc_rate - p_star

                print(f"{cond_name:<25} {ratio:>3} {esc_rate:>8.3f} {p_star:>6.3f} "
                      f"{implied_gap:>+6.3f} {avg_cost:>8.3f} {fp_rate:>7.3f} {fn_rate:>7.3f} "
                      f"{esc_acc:>7.3f} {n:>5}")

    # Summary table: avg across conditions per dataset × cost ratio
    print(f"\n{'='*70}")
    print("SUMMARY: Avg cost by dataset × cost ratio")
    print(f"{'='*70}")
    summary = df.groupby(["dataset", "cost_ratio"]).agg(
        esc_rate=("escalate", "mean"),
        avg_cost=("cost", "mean"),
        n=("escalate", "count"),
    ).reset_index()
    print(summary.to_string(index=False))

    print(f"\nAll results saved to {RESULTS_DIR}/")
