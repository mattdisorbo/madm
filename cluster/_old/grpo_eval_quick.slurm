#!/bin/bash
#SBATCH -J madm-grpo-eval
#SBATCH -o logs/grpo_eval_%j.out
#SBATCH -e logs/grpo_eval_%j.err
#SBATCH -p mi2101x
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:30:00

set -e
cd "${WORK}/madm"
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export HF_HOME=/work1/harangju/harang/.cache/huggingface
export UV_CACHE_DIR=/work1/harangju/harang/.cache/uv
export PIP_CACHE_DIR=/work1/harangju/harang/.cache/pip
export TMPDIR=/work1/harangju/harang/.cache/tmp
mkdir -p "$UV_CACHE_DIR" "$PIP_CACHE_DIR" "$TMPDIR"

uv pip install --reinstall torch --index-url https://download.pytorch.org/whl/rocm7.1
uv pip install --upgrade "transformers @ git+https://github.com/huggingface/transformers.git@main" typing_extensions datasets trl peft accelerate scikit-learn
uv pip install --reinstall torchvision --index-url https://download.pytorch.org/whl/rocm7.1

python -u << 'PYEOF'
import os, re, sys, torch, pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

MODEL = "Qwen/Qwen3.5-9B"
LORA_PATH = "outputs/grpo_lora/checkpoint-2000"
N = 50

# Load model
from transformers import AutoTokenizer, pipeline as hf_pipeline
try:
    from transformers import Qwen3_5ForCausalLM
    model = Qwen3_5ForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="auto")
except ImportError:
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="auto")

tokenizer = AutoTokenizer.from_pretrained(MODEL)
base_pipe = hf_pipeline("text-generation", model=model, tokenizer=tokenizer)

from peft import PeftModel
lora_model = PeftModel.from_pretrained(model, LORA_PATH)
lora_model.eval()
lora_pipe = hf_pipeline("text-generation", model=lora_model, tokenizer=tokenizer)

THOUGHT_PROMPT = (
    "TASK: Predict Person 1's decision. Explain your reasoning in 1 sentence. "
    "Then conclude with exactly 'PREDICTION: 1' or 'PREDICTION: 0'."
)
ESCALATE_TASK = (
    "An LLM was asked to predict Person 1's decision in this scenario.\n"
    "TASK: Respond with a digit ONLY.\n"
    "Output 0 to implement the LLM's prediction, or 1 to escalate to Person 1.\n"
    "A decision is considered correct if it equals Person 1's true decision."
)

def run(pipe, prompt, max_new=256):
    msgs = [{"role": "user", "content": prompt}]
    fmt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    out = pipe(fmt, max_new_tokens=max_new, return_full_text=False)
    text = out[0]["generated_text"]
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

def parse_pred(text):
    preds = re.findall(r'PREDICTION:\s*([01])', text)
    return int(preds[-1]) if preds else None

def parse_esc(text):
    m = re.search(r'[01]', text.strip())
    return int(m.group()) if m else None

def evaluate(pipe, label, scenarios, gts):
    results = []
    for scenario, gt in tqdm(zip(scenarios, gts), total=len(scenarios), desc=label):
        try:
            thought = run(base_pipe, f"{scenario}\n\n{THOUGHT_PROMPT}")
            pred = parse_pred(thought)
            if pred is None: continue
            esc_text = run(pipe, f"{scenario}\n\n{ESCALATE_TASK}", max_new=8)
            esc = parse_esc(esc_text)
            if esc is None: continue
        except Exception as e:
            print(f"  Error: {e}")
            continue
        results.append({"gt": gt, "pred": pred, "correct": pred==gt, "escalate": esc})
    return pd.DataFrame(results)

def print_results(df, label, ds_name):
    if len(df) == 0:
        print(f"  {ds_name}: No valid results!")
        return
    correct = df[df["correct"]]
    wrong = df[~df["correct"]]
    esc_r = correct["escalate"].mean() if len(correct) > 0 else float('nan')
    esc_w = wrong["escalate"].mean() if len(wrong) > 0 else float('nan')
    print(f"  {ds_name} ({label}): n={len(df)} acc={df['correct'].mean():.1%} esc={df['escalate'].mean():.1%} esc|R={esc_r:.1%}({len(correct)}) esc|W={esc_w:.1%}({len(wrong)}) W-R={esc_w-esc_r:+.1%}")

# === LendingClub ===
acc = pd.read_csv("data/accepted_10k.csv")
rej = pd.read_csv("data/rejected_10k.csv")
acc_norm = pd.DataFrame({'loan_amnt': acc['loan_amnt'], 'purpose': acc['purpose'],
    'emp_length': acc['emp_length'], 'dti': acc['dti'], 'fico': acc['fico_range_low'], 'accepted': 1})
rej_dti = rej['Debt-To-Income Ratio'].astype(str).str.replace('%','',regex=False)
rej_norm = pd.DataFrame({'loan_amnt': rej['Amount Requested'], 'purpose': rej['Loan Title'],
    'emp_length': rej['Employment Length'], 'dti': pd.to_numeric(rej_dti, errors='coerce'),
    'fico': pd.to_numeric(rej['Risk_Score'], errors='coerce'), 'accepted': 0})
lc = pd.concat([acc_norm, rej_norm], ignore_index=True).dropna(subset=['loan_amnt','dti','fico']).reset_index(drop=True)
_, lc_idx = train_test_split(lc.index, test_size=0.2, random_state=42)
lc_hold = lc.loc[lc_idx].sample(n=N, random_state=42)
lc_scenarios = [
    f"You are helping Person 1, a loan officer, decide whether to approve a loan application. "
    f"The applicant is requesting ${r['loan_amnt']:,.0f} for {r['purpose']}. "
    f"They have been employed for {r['emp_length']}, "
    f"a debt-to-income ratio of {r['dti']:.1f}%, and a credit score of {r['fico']:.0f}."
    for _, r in lc_hold.iterrows()
]
lc_gts = [int(r['accepted']) for _, r in lc_hold.iterrows()]

# === MoralMachine ===
sys.path.insert(0, "scripts")
from run_v2_MoralMachine import create_prompt_base as mm_prompt
mm = pd.read_csv("data/MoralMachine/SharedResponsesSurveyUSA1M.csv")
_, mm_idx = train_test_split(mm.index, test_size=0.2, random_state=42)
mm_hold = mm.loc[mm_idx].sample(n=N, random_state=42)
mm_scenarios = [mm_prompt(r) for _, r in mm_hold.iterrows()]
mm_gts = [int(r['Saved_0']) for _, r in mm_hold.iterrows()]

# Run all
for ds_name, scenarios, gts in [("LendingClub", lc_scenarios, lc_gts), ("MoralMachine", mm_scenarios, mm_gts)]:
    print(f"\n{'='*50}")
    print(f"  {ds_name} (n={N})")
    print(f"{'='*50}")
    for label, pipe in [("baseline", base_pipe), ("grpo", lora_pipe)]:
        df = evaluate(pipe, f"{ds_name}-{label}", scenarios, gts)
        print_results(df, label, ds_name)

print("\nDone!")
PYEOF
