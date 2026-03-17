"""Study 3: Isolated escalation — predict and escalate in two independent calls."""
import os, re, datetime, sys
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import openai

DATASET = os.environ.get("DATASET", "HotelBookings")
MODEL = os.environ.get("MODEL", "Qwen/Qwen3.5-9B")
PROVIDER = os.environ.get("PROVIDER", "together")
N_PER_CONDITION = int(os.environ.get("N_PER_CONDITION", "250"))
WORKERS = int(os.environ.get("WORKERS", "20"))
NOHINT = os.environ.get("NOHINT", "0") == "1"
COST_RATIO = os.environ.get("COST_RATIO", "")
OUTPUT_DIR = "results/study3"

if PROVIDER == "openai":
    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
else:
    client = openai.OpenAI(
        api_key=os.environ["TOGETHER_API_KEY"],
        base_url="https://api.together.xyz/v1",
    )


def llm(messages, max_tokens=512):
    import time as _time
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    kwargs = dict(model=MODEL, messages=messages, max_tokens=max_tokens)
    if PROVIDER == "together" and "Qwen3.5" in MODEL:
        kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
    for attempt in range(5):
        try:
            r = client.chat.completions.create(**kwargs)
            msg = r.choices[0].message
            text = (msg.content or "").strip()
            text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
            return text
        except Exception as e:
            if attempt == 4:
                raise
            wait = 2 ** attempt + 1
            _time.sleep(wait)


def parse_prediction(text):
    preds = re.findall(r'PREDICTION:\s*([01])', text)
    if preds:
        return int(preds[-1])
    return None


def parse_escalation(text):
    decisions = re.findall(r'DECISION:\s*([01])', text)
    if decisions:
        return int(decisions[-1])
    m = re.search(r'[01]', text.strip())
    return int(m.group()) if m else None


# Isolated escalation prompts: same info but no "You predicted" framing
ESC_PROMPTS = {
    "HotelBookings": (
        "A model was asked to predict Person 1's booking decision for the scenario above. "
        "The model predicted: {pred_label}.\n"
        "TASK: Should we implement this prediction, or escalate to Person 1?\n"
        "A decision is considered correct if we implement when the prediction matches "
        "Person 1's true decision, or escalate when the prediction does not.\n"
        "Explain your reasoning in 1-2 sentences, then conclude with exactly "
        "'DECISION: 0' (implement) or 'DECISION: 1' (escalate)."
    ),
    "MoralMachine": (
        "A model was asked to predict Person 1's decision for the scenario above. "
        "The model predicted: {pred_label}.\n"
        "TASK: Should we implement this prediction, or escalate to Person 1?\n"
        "A decision is considered correct if we implement when the prediction matches "
        "Person 1's true decision, or escalate when the prediction does not.\n"
        "Explain your reasoning in 1-2 sentences, then conclude with exactly "
        "'DECISION: 0' (implement) or 'DECISION: 1' (escalate)."
    ),
    "LendingClub": (
        "A model was asked to predict Person 1's decision for the scenario above. "
        "The model predicted: {pred_label}.\n"
        "TASK: Should we implement this prediction, or escalate to Person 1?\n"
        "A decision is considered correct if we implement when the prediction matches "
        "Person 1's true decision, or escalate when the prediction does not.\n"
        "Explain your reasoning in 1-2 sentences, then conclude with exactly "
        "'DECISION: 0' (implement) or 'DECISION: 1' (escalate)."
    ),
    "WikipediaToxicity": (
        "A model was asked to predict the crowd-workers' aggregate decision for the scenario above. "
        "The model predicted: {pred_label}.\n"
        "TASK: Should we implement this prediction, or escalate to the crowd-workers?\n"
        "A decision is considered correct if we implement when the prediction matches "
        "the crowd-workers' true aggregate decision, or escalate when the prediction does not.\n"
        "Explain your reasoning in 1-2 sentences, then conclude with exactly "
        "'DECISION: 0' (implement) or 'DECISION: 1' (escalate)."
    ),
    "MovieLens": (
        "A model was asked to predict which movie Person 1 would rate higher for the scenario above. "
        "The model predicted: {pred_label}.\n"
        "TASK: Should we implement this prediction, or escalate to Person 1?\n"
        "A decision is considered correct if we implement when the prediction matches "
        "the movie Person 1 would actually rate higher, or escalate when the prediction does not.\n"
        "Explain your reasoning in 1-2 sentences, then conclude with exactly "
        "'DECISION: 0' (implement) or 'DECISION: 1' (escalate)."
    ),
}

PRED_LABELS = {
    "HotelBookings": {1: "booking will be kept", 0: "booking will be canceled"},
    "MoralMachine": {1: "save the passengers", 0: "save the pedestrians"},
    "LendingClub": {1: "approve", 0: "reject"},
    "WikipediaToxicity": {1: "toxic", 0: "not toxic"},
    "MovieLens": {1: "Movie A", 0: "Movie B"},
}


def process_sample(scenario, gt, hint, predict_prompt, esc_template, pred_labels, condition_name):
    try:
        # Call 1: predict (single-turn)
        if NOHINT:
            pred_prompt = f"{scenario}\n\n{predict_prompt}"
        else:
            pred_prompt = f"{scenario}\n\n{hint}\n\n{predict_prompt}"

        thought = llm(pred_prompt)
        pred = parse_prediction(thought)
        if pred is None:
            print(f"  [FAIL] parse_prediction returned None. Output: {thought[:200]}", flush=True)
            return None

        # Call 2: escalate (independent single-turn, no prediction reasoning)
        pred_label = pred_labels.get(pred, str(pred))
        esc_prompt = esc_template.format(pred_label=pred_label)

        if COST_RATIO:
            ratio = int(COST_RATIO)
            esc_prompt = (
                f"Implementing a wrong answer costs {ratio}x more than escalating.\n\n"
                f"{esc_prompt}"
            )

        if NOHINT:
            esc_full = f"{scenario}\n\n{esc_prompt}"
        else:
            esc_full = f"{scenario}\n\n{hint}\n\n{esc_prompt}"

        esc_text = llm(esc_full)
        esc = parse_escalation(esc_text)
        if esc is None:
            print(f"  [FAIL] parse_escalation returned None. Output: {esc_text[:200]}", flush=True)
            return None

        return {
            "condition": condition_name,
            "ground_truth": gt,
            "prediction": pred,
            "correct": int(pred == gt),
            "escalate": esc,
            "prompt": pred_prompt,
            "esc_prompt": esc_full,
            "thought": thought,
            "esc_reasoning": esc_text,
            "timestamp": datetime.datetime.now().isoformat(),
        }
    except Exception as e:
        print(f"  [ERROR] {type(e).__name__}: {e}", flush=True)
        return None


if __name__ == "__main__":
    # Import dataset registry from study3
    sys.path.insert(0, os.path.dirname(__file__))
    from study3 import DATASETS

    if DATASET not in DATASETS:
        print(f"Unknown dataset: {DATASET}. Choose from: {list(DATASETS.keys())}")
        exit(1)

    ds = DATASETS[DATASET]
    esc_template = ESC_PROMPTS[DATASET]
    pred_labels = PRED_LABELS[DATASET]
    model_short = MODEL.split("/")[-1]

    print(f"Dataset: {DATASET}")
    print(f"Model: {MODEL}")
    print(f"Mode: isolated (two independent single-turn calls)")
    print(f"N per condition: {N_PER_CONDITION}")
    print(f"Workers: {WORKERS}")

    print(f"Loading {DATASET} data...")
    df = ds["load"]()
    conditions = ds["conditions"](df)
    gt_col = ds["gt_col"]
    predict_prompt = ds["predict_prompt"]
    create_prompt = ds["prompt"]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    summary_rows = []

    for cond in conditions:
        name = cond["name"]
        mask = cond["mask"]
        hint = cond["hint"]
        base_rate = cond["base_rate"]

        cost_tag = f"_cost{COST_RATIO}" if COST_RATIO else ""
        hint_tag = "_nohint" if NOHINT else ""
        out_path = f"{OUTPUT_DIR}/{DATASET}_{name}_isolated_nothink{cost_tag}{hint_tag}_{model_short}.csv"
        if os.path.exists(out_path):
            print(f"\n  Skipping {name} (already exists: {out_path})")
            continue

        subset = df[mask]
        sample = subset.sample(n=min(N_PER_CONDITION, len(subset)), random_state=42)
        scenarios = [create_prompt(r) for _, r in sample.iterrows()]
        gts = [int(r[gt_col]) for _, r in sample.iterrows()]

        print(f"\n{'='*60}")
        print(f"  {name} (base_rate={base_rate:.0%}, n={len(sample)})")
        print(f"{'='*60}")

        results = []
        failed = 0
        with ThreadPoolExecutor(max_workers=WORKERS) as executor:
            futures = {
                executor.submit(
                    process_sample, s, g, hint, predict_prompt,
                    esc_template, pred_labels, name,
                ): i
                for i, (s, g) in enumerate(zip(scenarios, gts))
            }
            for f in tqdm(as_completed(futures), total=len(futures), desc=name):
                result = f.result()
                if result:
                    results.append(result)
                else:
                    failed += 1

        rdf = pd.DataFrame(results)
        rdf.to_csv(out_path, index=False)

        if len(rdf) == 0:
            print(f"  No valid results!")
            continue

        pred_acc = rdf["correct"].mean()
        tp = ((rdf["escalate"] == 1) & (rdf["correct"] == 0)).sum()
        tn = ((rdf["escalate"] == 0) & (rdf["correct"] == 1)).sum()
        fp = ((rdf["escalate"] == 1) & (rdf["correct"] == 1)).sum()
        fn = ((rdf["escalate"] == 0) & (rdf["correct"] == 0)).sum()
        esc_acc = (tp + tn) / len(rdf)
        esc_rate = rdf["escalate"].mean()
        wrong = rdf[rdf["correct"] == 0]
        right = rdf[rdf["correct"] == 1]
        esc_w = wrong["escalate"].mean() if len(wrong) > 0 else float('nan')
        esc_r = right["escalate"].mean() if len(right) > 0 else float('nan')
        gap = pred_acc - esc_acc

        print(f"  n={len(rdf)} failed={failed}")
        print(f"  Pred acc:  {pred_acc:.1%}")
        print(f"  Esc acc:   {esc_acc:.1%} (TP={tp} TN={tn} FP={fp} FN={fn})")
        print(f"  Esc rate:  {esc_rate:.1%}")
        print(f"  Esc|W:     {esc_w:.1%} ({len(wrong)})")
        print(f"  Esc|R:     {esc_r:.1%} ({len(right)})")
        print(f"  Gap:       {gap:+.1%}")

        summary_rows.append({
            "condition": name,
            "base_rate": base_rate,
            "n": len(rdf),
            "pred_acc": pred_acc,
            "esc_acc": esc_acc,
            "esc_rate": esc_rate,
            "esc_w": esc_w,
            "esc_r": esc_r,
            "gap": gap,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        })

    summary = pd.DataFrame(summary_rows)
    cost_tag = f"_cost{COST_RATIO}" if COST_RATIO else ""
    hint_tag = "_nohint" if NOHINT else ""
    summary.to_csv(f"{OUTPUT_DIR}/{DATASET}_summary_isolated_nothink{cost_tag}{hint_tag}_{model_short}.csv", index=False)
    print(f"\n\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    if len(summary) > 0:
        print(summary[["condition", "base_rate", "pred_acc", "esc_acc", "esc_rate", "esc_w", "esc_r", "gap"]].to_string(index=False))
    print(f"\nSaved to {OUTPUT_DIR}/")
