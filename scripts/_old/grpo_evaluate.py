"""
GRPO Evaluation: LendingClub only.

Runs the 2-call pipeline on holdout data:
1. Predict (frozen, no LoRA) -> get prediction
2. Escalate (LoRA or baseline) -> get decision
3. Compute calibration gap: P(escalate|wrong) - P(escalate|right)

Usage:
    python scripts/grpo_evaluate.py [MODEL] [--no-lora] [--lora-path PATH]
"""

import os, re, sys, random
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3.5-9B"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LORA_PATH = os.path.join(BASE_DIR, "../outputs/grpo_lora")
DATA_DIR = os.path.join(BASE_DIR, "../data")
RESULTS_DIR = os.path.join(BASE_DIR, "../results/grpo")
N_EVAL = 500

for i, arg in enumerate(sys.argv):
    if arg == "--lora-path" and i + 1 < len(sys.argv):
        LORA_PATH = sys.argv[i + 1]
    if arg == "--n-eval" and i + 1 < len(sys.argv):
        N_EVAL = int(sys.argv[i + 1])

USE_LORA = "--no-lora" not in sys.argv
os.makedirs(RESULTS_DIR, exist_ok=True)

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

print("Model loaded.", flush=True)

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


def run_pipe(pipe, prompt, max_new_tokens=2048):
    if "Qwen3.5" in MODEL:
        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
    elif "Qwen3" in MODEL:
        messages = [{"role": "system", "content": "/no_think"}, {"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
    else:
        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
    out = pipe(formatted, max_new_tokens=max_new_tokens, return_full_text=False)
    text = out[0]["generated_text"]
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


if __name__ == "__main__":
    # Load LendingClub holdout
    print("Loading LendingClub holdout...", flush=True)
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

    _, holdout_idx = train_test_split(data.index, test_size=0.2, random_state=42)
    holdout = data.loc[holdout_idx]

    if len(holdout) > N_EVAL:
        holdout = holdout.sample(n=N_EVAL, random_state=42)
    print(f"Evaluating on {len(holdout)} holdout samples", flush=True)

    label = "grpo" if USE_LORA else "baseline"
    results = []

    for idx, row in tqdm(holdout.iterrows(), total=len(holdout), desc=f"Eval ({label})"):
        scenario = (
            f"You are helping Person 1, a loan officer, decide whether to approve a loan application. "
            f"The applicant is requesting ${row['loan_amnt']:,.0f} for {row['purpose']}. "
            f"They have been employed for {row['emp_length']}, "
            f"a debt-to-income ratio of {row['dti']:.1f}%, "
            f"and a credit score of {row['fico']:.0f}."
        )
        gt = int(row['accepted'])

        # Step 1: Predict (frozen)
        try:
            thought = run_pipe(predict_pipe, f"{scenario}\n\n{THOUGHT_PROMPT}")
            pred = parse_prediction(thought)
        except Exception as e:
            continue
        if pred is None:
            continue

        # Step 2: Escalate (LoRA or baseline)
        try:
            esc_response = run_pipe(escalate_pipe, f"{scenario}\n\n{ESCALATE_ONLY_TASK}", max_new_tokens=8)
            escalate = parse_decision(esc_response)
        except Exception as e:
            continue
        if escalate is None:
            continue

        results.append({
            "ground_truth": gt,
            "prediction": pred,
            "pred_correct": pred == gt,
            "escalate": escalate,
        })

    df = pd.DataFrame(results)
    results_path = os.path.join(RESULTS_DIR, f"eval_{label}_LendingClub.csv")
    df.to_csv(results_path, index=False)
    print(f"\nSaved to {results_path}", flush=True)

    # Calibration
    correct = df[df["pred_correct"]]
    wrong = df[~df["pred_correct"]]
    esc_right = correct["escalate"].mean() if len(correct) > 0 else float('nan')
    esc_wrong = wrong["escalate"].mean() if len(wrong) > 0 else float('nan')
    gap = esc_wrong - esc_right

    print(f"\n{'='*50}", flush=True)
    print(f"RESULTS ({label}) - LendingClub", flush=True)
    print(f"{'='*50}", flush=True)
    print(f"  n = {len(df)}", flush=True)
    print(f"  Accuracy: {df['pred_correct'].mean():.3f}", flush=True)
    print(f"  Escalation rate: {df['escalate'].mean():.3f}", flush=True)
    print(f"  P(esc|right): {esc_right:.3f} (n={len(correct)})", flush=True)
    print(f"  P(esc|wrong):  {esc_wrong:.3f} (n={len(wrong)})", flush=True)
    print(f"  Calibration gap: {gap:+.3f}", flush=True)
