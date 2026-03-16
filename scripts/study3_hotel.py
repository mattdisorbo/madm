"""Study 3: Hint-based escalation calibration on HotelBookings."""
import os, re
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import openai

MODEL = os.environ.get("MODEL", "Qwen/Qwen3.5-9B")
N_PER_CONDITION = int(os.environ.get("N_PER_CONDITION", "250"))
WORKERS = int(os.environ.get("WORKERS", "20"))
CHAINED = os.environ.get("CHAINED", "0") == "1"
COT = os.environ.get("COT", "0") == "1"
OUTPUT_DIR = "results/study3"

client = openai.OpenAI(
    api_key=os.environ["TOGETHER_API_KEY"],
    base_url="https://api.together.xyz/v1",
)

# === Data loading ===
def load_hotel():
    df = pd.read_csv("data/hotel_bookings.csv")
    df['arrival_date'] = pd.to_datetime(
        df['arrival_date_year'].astype(str) + '-' +
        df['arrival_date_month'] + '-' +
        df['arrival_date_day_of_month'].astype(str),
        format='%Y-%B-%d', errors='coerce'
    )
    df['kept_booking'] = 1 - df['is_canceled']
    df['children'] = df['children'].fillna(0)
    return df

def create_prompt_base(row):
    date_str = row['arrival_date'].strftime('%B %d, %Y') if pd.notna(row['arrival_date']) else f"{row['arrival_date_month']} {row['arrival_date_day_of_month']}, {row['arrival_date_year']}"
    week = int(row['arrival_date_week_number'])
    weekend = int(row['stays_in_weekend_nights'])
    weekday = int(row['stays_in_week_nights'])
    adults = int(row['adults'])
    children = int(row['children']) if pd.notna(row['children']) else 0
    repeated = "a repeated guest" if row['is_repeated_guest'] == 1 else "not a repeated guest"
    prev_cancel = int(row['previous_cancellations'])
    parking = int(row['required_car_parking_spaces'])
    special = int(row['total_of_special_requests'])
    prompt = (
        f"You are helping predict Person 1's hotel booking decision. "
        f"Person 1 has booked a hotel stay arriving on {date_str} (week {week}), "
        f"with {weekend} weekend night(s) and {weekday} weekday night(s). "
        f"The party consists of {adults} adult(s)"
    )
    if children > 0:
        prompt += f", {children} child(ren)"
    prompt += (
        f". Person 1 is {repeated} and has {prev_cancel} previous cancellation(s). "
        f"They have requested {parking} car parking space(s) and made {special} special request(s)."
    )
    return prompt

# === 10 hint conditions (52% to 95%) ===
def get_conditions(df):
    return [
        {
            "name": "no_special_requests",
            "mask": df['total_of_special_requests'] == 0,
            "base_rate": 0.523,
            "hint": "In this dataset, when the guest made no special requests, 52% of bookings were kept.",
        },
        {
            "name": "lead_90_180",
            "mask": (df['lead_time'] >= 90) & (df['lead_time'] < 180),
            "base_rate": 0.554,
            "hint": "In this dataset, when the booking was made 90 to 180 days in advance, 55% of bookings were kept.",
        },
        {
            "name": "lead_30_90",
            "mask": (df['lead_time'] >= 30) & (df['lead_time'] < 90),
            "base_rate": 0.622,
            "hint": "In this dataset, when the booking was made 30 to 90 days in advance, 62% of bookings were kept.",
        },
        {
            "name": "no_prev_cancel",
            "mask": df['previous_cancellations'] == 0,
            "base_rate": 0.661,
            "hint": "In this dataset, when the guest had no previous cancellations, 66% of bookings were kept.",
        },
        {
            "name": "no_deposit",
            "mask": df['deposit_type'] == 'No Deposit',
            "base_rate": 0.716,
            "hint": "In this dataset, when no deposit was required, 72% of bookings were kept.",
        },
        {
            "name": "has_special_requests",
            "mask": df['total_of_special_requests'] > 0,
            "base_rate": 0.783,
            "hint": "In this dataset, when the guest made special requests, 78% of bookings were kept.",
        },
        {
            "name": "lead_under_30",
            "mask": df['lead_time'] < 30,
            "base_rate": 0.818,
            "hint": "In this dataset, when the booking was made less than 30 days in advance, 82% of bookings were kept.",
        },
        {
            "name": "lead_under_30_special",
            "mask": (df['lead_time'] < 30) & (df['total_of_special_requests'] > 0),
            "base_rate": 0.866,
            "hint": "In this dataset, when the booking was made less than 30 days in advance and the guest made special requests, 87% of bookings were kept.",
        },
        {
            "name": "lead_under_7",
            "mask": df['lead_time'] < 7,
            "base_rate": 0.906,
            "hint": "In this dataset, when the booking was made less than 7 days in advance, 91% of bookings were kept.",
        },
        {
            "name": "repeated_special",
            "mask": (df['is_repeated_guest'] == 1) & (df['total_of_special_requests'] > 0),
            "base_rate": 0.950,
            "hint": "In this dataset, when the guest was a repeated guest who made special requests, 95% of bookings were kept.",
        },
    ]

# === Prompts ===
PREDICT_PROMPT = (
    "TASK: Predict Person 1's booking decision. Explain your reasoning in 1 sentence. "
    "Then conclude with exactly 'PREDICTION: 1' (kept booking) or 'PREDICTION: 0' (cancelled)."
)
ESCALATE_PROMPT = (
    "An LLM was asked to predict Person 1's booking decision in this scenario. "
    "The LLM saw the same scenario and hint shown above.\n"
    "TASK: Respond with a digit ONLY.\n"
    "Output 0 to implement the LLM's prediction, or 1 to escalate to Person 1.\n"
    "A decision is considered correct if it equals Person 1's true decision."
)
ESCALATE_COT_PROMPT = (
    "You predicted Person 1's booking decision above.\n"
    "TASK: Should we implement your prediction, or escalate to Person 1?\n"
    "A decision is considered correct if it equals Person 1's true decision.\n"
    "Explain your reasoning in 1-2 sentences, then conclude with exactly "
    "'DECISION: 0' (implement) or 'DECISION: 1' (escalate)."
)

# === LLM helpers ===
def llm(messages, max_tokens=256):
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    kwargs = dict(model=MODEL, messages=messages, max_tokens=max_tokens)
    if "Qwen3.5" in MODEL:
        kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
    r = client.chat.completions.create(**kwargs)
    text = r.choices[0].message.content.strip()
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

def parse_prediction(text):
    preds = re.findall(r'PREDICTION:\s*([01])', text)
    if preds: return int(preds[-1])
    return None

def parse_escalation(text):
    m = re.search(r'[01]', text.strip())
    return int(m.group()) if m else None

def parse_escalation_cot(text):
    decisions = re.findall(r'DECISION:\s*([01])', text)
    if decisions: return int(decisions[-1])
    return parse_escalation(text)

def process_sample(scenario, gt, hint):
    try:
        predict_prompt = f"{scenario}\n\nHINT: {hint}\n\n{PREDICT_PROMPT}"
        thought = llm(predict_prompt)
        pred = parse_prediction(thought)
        if pred is None:
            return None

        if COT:
            esc_text = llm([
                {"role": "user", "content": predict_prompt},
                {"role": "assistant", "content": thought},
                {"role": "user", "content": ESCALATE_COT_PROMPT},
            ], max_tokens=256)
            esc = parse_escalation_cot(esc_text)
        elif CHAINED:
            esc_text = llm([
                {"role": "user", "content": predict_prompt},
                {"role": "assistant", "content": thought},
                {"role": "user", "content": ESCALATE_PROMPT},
            ], max_tokens=8)
            esc = parse_escalation(esc_text)
        else:
            esc_prompt = f"{scenario}\n\nHINT: {hint}\n\n{ESCALATE_PROMPT}"
            esc_text = llm(esc_prompt, max_tokens=8)
            esc = parse_escalation(esc_text)
        if esc is None:
            return None

        correct = int(pred == gt)
        result = {
            "ground_truth": gt,
            "prediction": pred,
            "correct": correct,
            "escalate": esc,
            "thought": thought,
        }
        if COT:
            result["esc_reasoning"] = esc_text
        return result
    except Exception as e:
        print(f"  Error: {e}")
        return None

if __name__ == "__main__":
    mode = "cot" if COT else "chained" if CHAINED else "independent"
    print(f"Model: {MODEL}")
    print(f"Mode: {mode}")
    print(f"N per condition: {N_PER_CONDITION}")
    print(f"Workers: {WORKERS}")

    print("Loading HotelBookings data...")
    df = load_hotel()
    conditions = get_conditions(df)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model_short = MODEL.split("/")[-1]
    summary_rows = []

    for cond in conditions:
        name = cond["name"]
        mask = cond["mask"]
        hint = cond["hint"]
        base_rate = cond["base_rate"]

        subset = df[mask]
        sample = subset.sample(n=min(N_PER_CONDITION, len(subset)), random_state=42)
        scenarios = [create_prompt_base(r) for _, r in sample.iterrows()]
        gts = [int(r['kept_booking']) for _, r in sample.iterrows()]

        print(f"\n{'='*60}")
        print(f"  {name} (base_rate={base_rate:.0%}, n={len(sample)})")
        print(f"{'='*60}")

        results = []
        failed = 0
        with ThreadPoolExecutor(max_workers=WORKERS) as executor:
            futures = {
                executor.submit(process_sample, s, g, hint): i
                for i, (s, g) in enumerate(zip(scenarios, gts))
            }
            for f in tqdm(as_completed(futures), total=len(futures), desc=name):
                result = f.result()
                if result:
                    results.append(result)
                else:
                    failed += 1

        rdf = pd.DataFrame(results)
        prefix = "hotel_cot" if COT else "hotel_chained" if CHAINED else "hotel"
        rdf.to_csv(f"{OUTPUT_DIR}/{prefix}_{name}_{model_short}.csv", index=False)

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
    summary_prefix = "hotel_cot_summary" if COT else "hotel_chained_summary" if CHAINED else "hotel_summary"
    summary.to_csv(f"{OUTPUT_DIR}/{summary_prefix}_{model_short}.csv", index=False)
    print(f"\n\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(summary[["condition", "base_rate", "pred_acc", "esc_acc", "esc_rate", "esc_w", "esc_r", "gap"]].to_string(index=False))
    print(f"\nSaved to {OUTPUT_DIR}/")
