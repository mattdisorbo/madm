"""
SFT for cost-sensitive escalation with CoT reasoning.

Trains on 4 datasets, holds out 1 for testing.
Target responses include explicit base rate reasoning.

Usage:
  python scripts/sft_escalation.py [--output-dir DIR] [--holdout MovieLens]
"""

import argparse
import glob
import os
import re
import sys
import random
import torch
import pandas as pd
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, PeftModel

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from study3 import (
    load_hotel, hotel_conditions,
    load_moral, moral_conditions,
    load_lending, lending_conditions,
    load_wiki, wiki_conditions,
    load_movielens, movielens_conditions,
)

COST_RATIOS = [2, 4, 8, 10, 20, 50]
RESULTS_DIR = "results/study3"

COST_FORMATS = {
    "original": "Cost ratio R = {R}. A wrong implementation costs {R}x more than escalating.",
    "dollar": "Escalation costs $1. A wrong implementation costs ${R}.",
    "wording": "The cost of being wrong is {R} times the cost of asking a human.",
    "study3": "Implementing a wrong answer costs {R}x more than escalating.",
}

BASE_ESC = (
    "You predicted the outcome above.\n"
    "TASK: Should we implement your prediction, or escalate to a human?\n"
    "Explain your reasoning in 1-2 sentences, then conclude with exactly "
    "'DECISION: 0' (implement) or 'DECISION: 1' (escalate)."
)

DATASET_REGISTRY = {
    "HotelBookings": {"load": load_hotel, "conditions": hotel_conditions},
    "MoralMachine": {"load": load_moral, "conditions": moral_conditions},
    "LendingClub": {"load": load_lending, "conditions": lending_conditions},
    "WikipediaToxicity": {"load": load_wiki, "conditions": wiki_conditions},
    "MovieLens": {"load": load_movielens, "conditions": movielens_conditions},
}


def make_esc_prompt(R, fmt):
    cost_line = COST_FORMATS[fmt].format(R=R)
    return f"{cost_line}\n\n{BASE_ESC}"


def make_cot_response(R, base_rate, optimal):
    error_rate = 1 - base_rate
    expected_cost = R * error_rate
    if optimal == 1:
        return (
            f"The hint suggests an accuracy of {base_rate:.0%}, "
            f"so the error rate is {error_rate:.0%}. "
            f"The expected cost of implementing is {R} * {error_rate:.2f} = {expected_cost:.2f}, "
            f"which exceeds 1. Escalating is the better choice. "
            f"DECISION: 1"
        )
    else:
        return (
            f"The hint suggests an accuracy of {base_rate:.0%}, "
            f"so the error rate is {error_rate:.0%}. "
            f"The expected cost of implementing is {R} * {error_rate:.2f} = {expected_cost:.2f}, "
            f"which is below 1. Implementing is the better choice. "
            f"DECISION: 0"
        )


def build_dataset(tokenizer, holdout, n_per_condition=5, seed=42):
    """Build SFT dataset from study3 CSVs across multiple datasets."""
    random.seed(seed)

    train_datasets = [name for name in DATASET_REGISTRY if name != holdout]
    print(f"Training datasets: {train_datasets}", flush=True)
    print(f"Holdout dataset: {holdout}", flush=True)

    rows = []
    for ds_name in train_datasets:
        ds = DATASET_REGISTRY[ds_name]
        print(f"\nLoading {ds_name}...", flush=True)

        try:
            df_full = ds["load"]()
            conditions = ds["conditions"](df_full)
        except Exception as e:
            print(f"  Failed to load {ds_name}: {e}", flush=True)
            continue

        ds_rows = 0
        for cond in conditions:
            name = cond["name"]
            base_rate = cond["base_rate"]

            csv = f"{RESULTS_DIR}/{ds_name}_{name}_nothink_Qwen3.5-9B.csv"
            if not os.path.exists(csv):
                print(f"  Skipping {ds_name}/{name}: no CSV", flush=True)
                continue

            df = pd.read_csv(csv)
            sample = df.head(n_per_condition)

            for _, row in sample.iterrows():
                prompt_text = row["prompt"]
                thought = str(row["thought"])

                for R in COST_RATIOS:
                    optimal = 1 if R * (1 - base_rate) > 1 else 0
                    response = make_cot_response(R, base_rate, optimal)

                    for fmt in COST_FORMATS:
                        esc_prompt = make_esc_prompt(R, fmt)

                        messages = [
                            {"role": "user", "content": prompt_text},
                            {"role": "assistant", "content": thought},
                            {"role": "user", "content": esc_prompt},
                            {"role": "assistant", "content": response},
                        ]

                        text = tokenizer.apply_chat_template(
                            messages, tokenize=False,
                            add_generation_prompt=False,
                            enable_thinking=False,
                        )

                        rows.append({
                            "text": text,
                            "dataset": ds_name,
                            "condition": name,
                            "base_rate": base_rate,
                            "cost_ratio": R,
                            "format": fmt,
                            "optimal": optimal,
                        })
                        ds_rows += 1

        print(f"  {ds_name}: {ds_rows} pairs", flush=True)

    random.shuffle(rows)

    n_esc = sum(r["optimal"] == 1 for r in rows)
    print(f"\nBuilt {len(rows)} total examples ({n_esc} escalate, {len(rows)-n_esc} implement)", flush=True)
    print(f"Formats: {list(COST_FORMATS.keys())}", flush=True)
    print(f"Escalation rate by R:", flush=True)
    for R in COST_RATIOS:
        at_r = [r for r in rows if r["cost_ratio"] == R]
        if at_r:
            esc = sum(r["optimal"] == 1 for r in at_r)
            print(f"  R={R:>2}: {esc}/{len(at_r)} escalate ({esc/len(at_r):.0%})", flush=True)

    print(f"\nExamples by dataset:", flush=True)
    for ds_name in train_datasets:
        ds_count = sum(1 for r in rows if r["dataset"] == ds_name)
        print(f"  {ds_name}: {ds_count}", flush=True)

    return Dataset.from_list(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--output-dir", default="outputs/sft_esc")
    parser.add_argument("--holdout", default="MovieLens")
    parser.add_argument("--n", type=int, default=5)
    args = parser.parse_args()

    print(f"Model: {args.model}", flush=True)
    print(f"Holdout: {args.holdout}", flush=True)

    if args.holdout not in DATASET_REGISTRY:
        print(f"Unknown holdout: {args.holdout}. Choose from: {list(DATASET_REGISTRY.keys())}")
        sys.exit(1)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = build_dataset(tokenizer, holdout=args.holdout, n_per_condition=args.n)
    print(f"Dataset: {len(dataset)} examples", flush=True)

    split = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    print(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}", flush=True)

    peft_config = LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules="all-linear",
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    config = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        num_train_epochs=1,
        learning_rate=5e-5,
        logging_steps=10,
        save_steps=25,
        save_total_limit=2,
        eval_steps=50,
        eval_strategy="steps",
        bf16=True,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        report_to="none",
        max_length=768,
    )

    print(f"Loading model {args.model}...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)

    # Qwen3.5 mm_token_type_ids fix
    _orig_validate = model._validate_model_kwargs
    def _patched_validate(model_kwargs):
        model_kwargs.pop("mm_token_type_ids", None)
        return _orig_validate(model_kwargs)
    model._validate_model_kwargs = _patched_validate

    # Check for checkpoint to resume from
    resume_from = None
    if os.path.isdir(args.output_dir):
        checkpoints = sorted(
            glob.glob(os.path.join(args.output_dir, "checkpoint-*")),
            key=lambda x: int(os.path.basename(x).split("-")[1]),
        )
        if checkpoints:
            resume_from = checkpoints[-1]
            print(f"Resuming from checkpoint: {resume_from}", flush=True)

    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    print("Starting SFT training...", flush=True)
    trainer.train(resume_from_checkpoint=resume_from)

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
