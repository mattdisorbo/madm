"""Neural activation auditor - API version.

Tests whether self-critique improves loan acceptance decisions using Claude API.
Cannot perform activation steering (no layer access), but tests reasoning paths.
"""

import os
import re
import pandas as pd
from openai import OpenAI

# ======================== CONFIG ========================

N_SAMPLES = 50
ACCEPTED_CSV = "data/accepted_10k.csv"
REJECTED_CSV = "data/rejected_10k.csv"

# ======================== LOAD DATA ========================

df_accepted = pd.read_csv(ACCEPTED_CSV)
df_rejected = pd.read_csv(REJECTED_CSV)

df = pd.concat(
    [
        df_accepted[["loan_amnt", "title", "addr_state", "emp_length", "policy_code"]]
        .assign(accepted=1),
        df_rejected.rename(
            columns={
                "Amount Requested": "loan_amnt",
                "Loan Title": "title",
                "State": "addr_state",
                "Employment Length": "emp_length",
                "Policy Code": "policy_code",
            }
        )[["loan_amnt", "title", "addr_state", "emp_length", "policy_code"]]
        .assign(accepted=0),
    ],
    ignore_index=True,
)

df["title"] = df["title"].str.lower().str.replace("_", " ", regex=False)
df["emp_length"] = df["emp_length"].map(
    {
        "< 1 year": 0,
        "1 year": 1,
        "2 years": 2,
        "3 years": 3,
        "4 years": 4,
        "5 years": 5,
        "6 years": 6,
        "7 years": 7,
        "8 years": 8,
        "9 years": 9,
        "10+ years": 10,
    }
)

print(f"Loaded {len(df)} total loan records")

# ======================== API SETUP ========================

client = OpenAI()

def call_llm(prompt: str, max_tokens: int = 100) -> str:
    """Call OpenAI API with a prompt."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # Cheap and fast - use "gpt-4o" for better quality
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

# ======================== PROMPT BUILDER ========================

def create_prompt_base(row):
    loan_amnt = int(round(row["loan_amnt"]))
    emp_length = int(round(row["emp_length"]))

    base = (
        f"Someone from {row['addr_state']} is applying for a loan of "
        f"${loan_amnt}."
    )

    if pd.notna(row["title"]):
        base += f" The loan is for {row['title']}."

    base += f" They have been employed for {emp_length} years."

    return base

# ======================== REASONING PATHS ========================

def get_base_support_decision(scenario: str) -> dict:
    """Base path: initial decision + supporting argument + final decision."""

    # Initial decision
    pred_prompt = (
        f"{scenario}\n\n"
        "Accept or reject the loan. Respond with only one word: accept or reject.\n"
        "Decision:"
    )
    prediction = call_llm(pred_prompt, max_tokens=10)

    # Supporting argument
    support_prompt = (
        f"SCENARIO: {scenario}\n"
        f"INITIAL DECISION: {prediction}\n\n"
        "Provide one reason why this decision is CORRECT.\n"
        "Keep it to one sentence starting with 'This is potentially because'"
    )
    support = call_llm(support_prompt, max_tokens=100)

    # Final decision
    final_prompt = (
        f"SCENARIO: {scenario}\n"
        f"INITIAL DECISION: {prediction}\n"
        f"SUPPORT OF INITIAL DECISION: {support}\n\n"
        "Final Decision - respond with only one word: accept or reject.\n"
        "Decision:"
    )
    final_text = call_llm(final_prompt, max_tokens=10)

    # Extract decision
    m = re.findall(r"\b(accept|reject)\b", final_text.lower())
    decision = m[-1] if m else None

    return {
        "decision": decision,
        "prediction": prediction,
        "support": support,
        "final_text": final_text
    }

def get_critique_decision(scenario: str) -> dict:
    """Auditor path: initial decision + critique + final decision."""

    # Initial decision
    pred_prompt = (
        f"{scenario}\n\n"
        "Accept or reject the loan. Respond with only one word: accept or reject.\n"
        "Decision:"
    )
    prediction = call_llm(pred_prompt, max_tokens=10)

    # Critique
    critique_prompt = (
        f"SCENARIO: {scenario}\n"
        f"INITIAL DECISION: {prediction}\n\n"
        "Provide one reason why this decision might be INCORRECT.\n"
        "Keep it to one sentence starting with 'On the other hand,'"
    )
    critique = call_llm(critique_prompt, max_tokens=100)

    # Final decision
    final_prompt = (
        f"SCENARIO: {scenario}\n"
        f"INITIAL DECISION: {prediction}\n"
        f"CRITIQUE OF INITIAL DECISION: {critique}\n\n"
        "Final Decision - respond with only one word: accept or reject.\n"
        "Decision:"
    )
    final_text = call_llm(final_prompt, max_tokens=10)

    # Extract decision
    m = re.findall(r"\b(accept|reject)\b", final_text.lower())
    decision = m[-1] if m else None

    return {
        "decision": decision,
        "prediction": prediction,
        "critique": critique,
        "final_text": final_text
    }

# ======================== COLLECTION LOOP ========================

results = []
print(f"\nStarting evaluation: {N_SAMPLES} samples...\n")

sample_count = 0
while sample_count < N_SAMPLES:
    row = df.sample(1).iloc[0]
    if pd.isna(row["emp_length"]):
        continue

    ground_truth = "accept" if row["accepted"] == 1 else "reject"
    scenario = create_prompt_base(row)

    try:
        base_res = get_base_support_decision(scenario)
        critique_res = get_critique_decision(scenario)

        if base_res["decision"] and critique_res["decision"]:
            sample_count += 1

            results.append({
                "ground_truth": ground_truth,
                "base_decision": base_res["decision"],
                "critique_decision": critique_res["decision"],
                "base_correct": base_res["decision"] == ground_truth,
                "critique_correct": critique_res["decision"] == ground_truth,
            })

            print(
                f"  Sample {sample_count}/{N_SAMPLES} | "
                f"Actual: {ground_truth:6} | Base: {base_res['decision']:6} | "
                f"Critique: {critique_res['decision']:6}"
            )
        else:
            print(f"  Skip (no clear decision) | Base: {base_res['final_text'][:30]} | "
                  f"Critique: {critique_res['final_text'][:30]}")
    except Exception as e:
        print(f"  Error on sample: {e}")
        continue

# ======================== RESULTS ========================

df_results = pd.DataFrame(results)

base_correct = df_results["base_correct"].sum()
critique_correct = df_results["critique_correct"].sum()

base_acc = (base_correct / N_SAMPLES) * 100
critique_acc = (critique_correct / N_SAMPLES) * 100

print("\n" + "=" * 60)
print("ACCURACY REPORT")
print("=" * 60)
print(f"  Base Accuracy (Support):   {base_acc:.1f}% ({base_correct}/{N_SAMPLES})")
print(f"  Critique Accuracy:         {critique_acc:.1f}% ({critique_correct}/{N_SAMPLES})")
print(f"  Accuracy Delta:            {critique_acc - base_acc:+.1f}%")
print("=" * 60)

# ======================== FLIPS ANALYSIS ========================

flips = df_results[df_results["base_decision"] != df_results["critique_decision"]]
corrective_flips = flips[
    (flips["critique_correct"] == True) & (flips["base_correct"] == False)
]
harmful_flips = flips[
    (flips["critique_correct"] == False) & (flips["base_correct"] == True)
]

print(f"\nFLIPS ANALYSIS")
print(f"  Total decision changes:    {len(flips)}/{N_SAMPLES} ({len(flips)/N_SAMPLES*100:.1f}%)")
print(f"  Corrective flips:          {len(corrective_flips)} (critique fixed base error)")
print(f"  Harmful flips:             {len(harmful_flips)} (critique broke base correct)")
print(f"  Neutral flips:             {len(flips) - len(corrective_flips) - len(harmful_flips)}")

print("\n" + "=" * 60)
print("CONCLUSION")
print("=" * 60)
if critique_acc > base_acc:
    print(f"✓ Self-critique IMPROVED accuracy by {critique_acc - base_acc:.1f}%")
elif critique_acc < base_acc:
    print(f"✗ Self-critique DECREASED accuracy by {base_acc - critique_acc:.1f}%")
else:
    print(f"= Self-critique had NO EFFECT on accuracy")
print("=" * 60)
