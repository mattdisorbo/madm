"""
Evaluate DPO model on real study3 prompts with various cost framings.

Uses actual Qwen3.5-9B predictions from study3 results.
Only replaces the escalation decision.
"""

import argparse
import glob
import os
import re
import sys
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from study3 import hotel_conditions, load_hotel, lending_conditions, moral_conditions, \
    wiki_conditions, movielens_conditions, HOTEL, LENDING, MORAL, WIKI, MOVIELENS

COST_RATIOS = [2, 4, 8, 10, 20, 50]

DATASETS = {
    "HotelBookings": {"load": load_hotel, "conditions": hotel_conditions, "config": HOTEL},
    "LendingClub": {"load": None, "conditions": lending_conditions, "config": LENDING},
    "MoralMachine": {"load": None, "conditions": moral_conditions, "config": MORAL},
    "WikipediaToxicity": {"load": None, "conditions": wiki_conditions, "config": WIKI},
    "MovieLens": {"load": None, "conditions": movielens_conditions, "config": MOVIELENS},
}


def parse_decision(text):
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    match = re.search(r'[01]', text)
    return int(match.group()) if match else None


def make_esc_prompt(R, fmt="original"):
    base_esc = (
        "You predicted the outcome above.\n"
        "TASK: Should we implement your prediction, or escalate to a human?\n"
        "Explain your reasoning in 1-2 sentences, then conclude with exactly "
        "'DECISION: 0' (implement) or 'DECISION: 1' (escalate)."
    )
    if fmt == "original":
        return f"Cost ratio R = {R}. A wrong implementation costs {R}x more than escalating.\n\n{base_esc}"
    elif fmt == "dollar":
        return f"Escalation costs $1. A wrong implementation costs ${R}.\n\n{base_esc}"
    elif fmt == "wording":
        return f"The cost of being wrong is {R} times the cost of asking a human.\n\n{base_esc}"
    elif fmt == "none":
        return base_esc
    elif fmt == "study3":
        return f"Implementing a wrong answer costs {R}x more than escalating.\n\n{base_esc}"


FORMATS = ["study3", "original", "dollar", "wording", "none"]


def eval_model(model, tokenizer, device, dataset_name, n_per_condition=10, fmt_list=None):
    """Evaluate on study3 data. Uses nothink CSVs for predictions."""
    results_dir = "results/study3"

    # Get conditions and base rates
    ds = DATASETS[dataset_name]
    try:
        if ds["load"]:
            df_full = ds["load"]()
        else:
            df_full = pd.DataFrame()  # some datasets load differently
        conditions = ds["conditions"](df_full) if len(df_full) > 0 else []
    except Exception:
        conditions = []

    # Build condition -> base_rate mapping from available CSVs
    cond_br = {}
    pattern = f"{results_dir}/{dataset_name}_*_nothink_Qwen3.5-9B.csv"
    csv_files = glob.glob(pattern)
    csv_files = [f for f in csv_files if "cost" not in f and "nohint" not in f and "summary" not in f and "isolated" not in f]

    if not csv_files:
        print(f"  No CSVs found for {dataset_name}", flush=True)
        return {}

    # Get base rates from conditions if available
    if conditions:
        for c in conditions:
            cond_br[c["name"]] = c["base_rate"]

    all_results = {}
    formats_to_test = fmt_list if fmt_list else FORMATS

    for fmt in formats_to_test:
        correct_by_r = {R: {"c": 0, "n": 0} for R in COST_RATIOS}
        total_correct = 0
        total_n = 0

        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            if len(df) == 0:
                continue

            # Get condition name from filename
            basename = os.path.basename(csv_file)
            cond_name = basename.replace(f"{dataset_name}_", "").replace("_nothink_Qwen3.5-9B.csv", "")
            base_rate = cond_br.get(cond_name, 0.5)

            # Sample rows
            sample = df.head(n_per_condition)

            for _, row in sample.iterrows():
                prompt_text = row["prompt"]
                thought = str(row["thought"])
                pred_correct = int(row["correct"])

                for R in COST_RATIOS:
                    optimal = 1 if R * (1 - base_rate) > 1 else 0
                    esc_prompt = make_esc_prompt(R, fmt)

                    # Multi-turn: user prompt, assistant prediction, user escalation
                    messages = [
                        {"role": "user", "content": prompt_text},
                        {"role": "assistant", "content": thought},
                        {"role": "user", "content": esc_prompt},
                    ]
                    text = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True,
                        enable_thinking=False,
                    )
                    inputs = tokenizer(text, return_tensors="pt").to(device)
                    with torch.no_grad():
                        out = model.generate(**inputs, max_new_tokens=64, do_sample=False)
                    gen = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

                    # Parse DECISION: 0 or DECISION: 1
                    decision_match = re.search(r'DECISION:\s*([01])', gen)
                    if decision_match:
                        decision = int(decision_match.group(1))
                    else:
                        decision = parse_decision(gen)

                    is_correct = decision == optimal
                    total_correct += int(is_correct)
                    total_n += 1
                    correct_by_r[R]["c"] += int(is_correct)
                    correct_by_r[R]["n"] += 1

        acc = total_correct / total_n if total_n > 0 else 0
        all_results[fmt] = acc
        print(f"  {fmt:<12}: {total_correct}/{total_n} ({acc:.1%})", flush=True)
        for R in COST_RATIOS:
            d = correct_by_r[R]
            if d["n"] > 0:
                print(f"    R={R:>2}: {d['c']/d['n']:.0%} ({d['n']})", flush=True)

    return all_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--n", type=int, default=10, help="Samples per condition")
    parser.add_argument("--dataset", default="HotelBookings")
    parser.add_argument("--formats", default=None, help="Comma-separated formats to test (default: all)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    print(f"Model: {args.model}", flush=True)
    print(f"Dataset: {args.dataset}", flush=True)
    print(f"N per condition: {args.n}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(device)
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
        print("Adapter loaded.", flush=True)
    model.eval()

    fmt_list = args.formats.split(",") if args.formats else None
    print("\n=== DPO-TRAINED ===" if args.adapter else "\n=== BASELINE ===", flush=True)
    eval_model(model, tokenizer, device, args.dataset, n_per_condition=args.n, fmt_list=fmt_list)

    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
