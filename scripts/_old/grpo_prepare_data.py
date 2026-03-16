"""
GRPO Data Preparation: LendingClub only.

Generates frozen-model predictions via Together API, then saves
training dataset for GRPO escalation training.

Usage:
    python scripts/grpo_prepare_data.py
"""

import os, re, random
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import openai

MODEL = "Qwen/Qwen3.5-9B"
TOGETHER_MODEL = "Qwen/Qwen3.5-9B"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "../data")
OUTPUT_DIR = os.path.join(BASE_DIR, "../data/grpo_dataset")

# Together API (OpenAI-compatible)
client = openai.OpenAI(
    api_key=os.environ["TOGETHER_API_KEY"],
    base_url="https://api.together.xyz/v1",
)

THOUGHT_PROMPT = (
    "TASK: Predict Person 1's decision. Explain your reasoning in 1 sentence. "
    "Then conclude with exactly 'PREDICTION: 1' (approve) or 'PREDICTION: 0' (reject)."
)

ESCALATE_ONLY_TASK = (
    "An LLM was asked to predict Person 1's decision in this scenario.\n"
    "TASK: Respond with a digit ONLY.\n"
    "Output 0 to implement the LLM's prediction, or 1 to escalate to Person 1.\n"
    "A decision is considered correct if it equals Person 1's true decision."
)


def llm(prompt):
    r = client.chat.completions.create(
        model=TOGETHER_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=256,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    text = r.choices[0].message.content.strip()
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    return text


def parse_prediction(text):
    preds = re.findall(r'PREDICTION:\s*([01])', text)
    if preds:
        return int(preds[-1])
    digits = re.findall(r'[01]', text.strip())
    if digits:
        return int(digits[0])
    return None


def create_prompt_base(row):
    return (
        f"You are helping Person 1, a loan officer, decide whether to approve a loan application. "
        f"The applicant is requesting ${row['loan_amnt']:,.0f} for {row['purpose']}. "
        f"They have been employed for {row['emp_length']}, "
        f"a debt-to-income ratio of {row['dti']:.1f}%, "
        f"and a credit score of {row['fico']:.0f}."
    )


def process_sample(row):
    """Generate prediction for one sample. Returns dict or None on failure."""
    scenario = create_prompt_base(row)
    gt = int(row['accepted'])

    try:
        response = llm(f"{scenario}\n\n{THOUGHT_PROMPT}")
        pred = parse_prediction(response)
    except Exception as e:
        return None

    if pred is None:
        return None

    escalation_prompt = f"{scenario}\n\n{ESCALATE_ONLY_TASK}"

    return {
        "prompt": escalation_prompt,
        "ground_truth": str(gt),
        "prediction": str(pred),
    }


if __name__ == "__main__":
    # Load LendingClub data
    print("Loading LendingClub data...", flush=True)
    acc = pd.read_csv(os.path.join(DATA_DIR, "accepted_10k.csv"))
    rej = pd.read_csv(os.path.join(DATA_DIR, "rejected_10k.csv"))

    acc_norm = pd.DataFrame({
        'loan_amnt': acc['loan_amnt'], 'purpose': acc['purpose'],
        'emp_length': acc['emp_length'], 'dti': acc['dti'],
        'fico': acc['fico_range_low'], 'accepted': 1,
    })
    rej_dti = rej['Debt-To-Income Ratio'].astype(str).str.replace('%', '', regex=False)
    rej_norm = pd.DataFrame({
        'loan_amnt': rej['Amount Requested'], 'purpose': rej['Loan Title'],
        'emp_length': rej['Employment Length'],
        'dti': pd.to_numeric(rej_dti, errors='coerce'),
        'fico': pd.to_numeric(rej['Risk_Score'], errors='coerce'),
        'accepted': 0,
    })
    data = pd.concat([acc_norm, rej_norm], ignore_index=True)
    data = data.dropna(subset=['loan_amnt', 'dti', 'fico']).reset_index(drop=True)

    # Sample balanced classes (limited by rejected count after dropna)
    n_rejected = len(data[data['accepted'] == 0])
    n_per_class = min(5000, n_rejected)
    accepted = data[data['accepted'] == 1].sample(n=n_per_class, random_state=42)
    rejected = data[data['accepted'] == 0].sample(n=n_per_class, random_state=42)
    train_data = pd.concat([accepted, rejected], ignore_index=True)
    print(f"Training samples: {len(train_data)} ({n_per_class} accepted, {n_per_class} rejected)", flush=True)

    # Generate predictions in parallel via Together API
    print("Generating predictions via Together API...", flush=True)
    examples = []
    failed = 0

    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(process_sample, train_data.loc[idx]): idx
                   for idx in train_data.index}
        for f in tqdm(as_completed(futures), total=len(futures), desc="Predictions"):
            result = f.result()
            if result is not None:
                examples.append(result)
            else:
                failed += 1

    print(f"Done: {len(examples)} ok, {failed} failed", flush=True)

    # Save as HF Dataset
    from datasets import Dataset
    ds = Dataset.from_list(examples)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ds.save_to_disk(OUTPUT_DIR)
    print(f"Saved dataset to {OUTPUT_DIR} ({len(ds)} examples)", flush=True)

    # Summary
    df = pd.DataFrame(examples)
    pred_correct = (df['prediction'] == df['ground_truth']).mean()
    print(f"Prediction accuracy: {pred_correct:.3f}", flush=True)
