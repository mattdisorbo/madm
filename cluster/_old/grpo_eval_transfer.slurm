#!/bin/bash
#SBATCH -J madm-grpo-xfer
#SBATCH -o logs/grpo_xfer_%j.out
#SBATCH -e logs/grpo_xfer_%j.err
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
import os, re, torch, pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

MODEL = "Qwen/Qwen3.5-9B"
LORA_PATH = "outputs/grpo_lora/checkpoint-2000"
N = 50

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
        print(f"  {ds_name} ({label}): No valid results!")
        return
    correct = df[df["correct"]]
    wrong = df[~df["correct"]]
    esc_r = correct["escalate"].mean() if len(correct) > 0 else float('nan')
    esc_w = wrong["escalate"].mean() if len(wrong) > 0 else float('nan')
    print(f"  {ds_name} ({label}): n={len(df)} acc={df['correct'].mean():.1%} esc={df['escalate'].mean():.1%} esc|R={esc_r:.1%}({len(correct)}) esc|W={esc_w:.1%}({len(wrong)}) W-R={esc_w-esc_r:+.1%}")

# === MoralMachine ===
education_map = {
    "underHigh": "Less Than a High School Diploma", "high": "High School Diploma",
    "vocational": "Vocational Training", "college": "Attended College",
    "bachelor": "Bachelor Degree", "graduate": "Graduate Degree", "other": "Other"
}
income_map = {
    "under5000": "Under $5,000", "5000": "$5,000-$10,000", "10000": "$10,001-$15,000",
    "15000": "$15,001-$25,000", "25000": "$25,001-$35,000", "35000": "$35,001-$50,000",
    "50000": "$50,001-$80,000", "80000": "$80,001-$100,000", "above100000": "Over $100,000"
}

def mm_prompt(row):
    def action(val): return "doesn't swerve" if val == 0 else "swerves"
    def group(val): return "passengers" if val == 1 else "pedestrians"
    def attr_text(attr):
        if attr == "Hoomans": attr = "Human"
        if attr == "Pets": attr = "pets (dog/cat)"
        attr = attr.lower()
        return "" if attr in ["more", "less", "rand"] else f"{attr} "
    def crossing_signal_text(val):
        if val == 1: return "legally crossing the street"
        elif val == 2: return "illegally crossing the street"
        else: return ""
    base = (
        f"An autonomous vehicle is about to get in an accident. "
        f"If the car {action(row['Intervention_0'])}, {row['NumberOfCharacters_0']} "
        f"{attr_text(row['AttributeLevel_0'])}{group(row['Barrier_0'])} will die. "
        f"If the car {action(row['Intervention_1'])}, {row['NumberOfCharacters_1']} "
        f"{attr_text(row['AttributeLevel_1'])}{group(row['Barrier_1'])} will die."
    )
    ped0, ped1 = row['PedPed_0'], row['PedPed_1']
    s0 = crossing_signal_text(row['CrossingSignal_0'])
    s1 = crossing_signal_text(row['CrossingSignal_1'])
    if ped0 == 1 and ped1 == 0 and s0: base += f" The pedestrians are {s0}."
    elif ped0 == 0 and ped1 == 1 and s1: base += f" The pedestrians are {s1}."
    elif ped0 == 1 and ped1 == 1:
        parts = []
        if s0: parts.append(f"The first group of pedestrians are {s0}.")
        if s1: parts.append(f"The second group of pedestrians are {s1}.")
        if parts: base += " " + " ".join(parts)
    return (base +
        " Person 1, with the following characteristics, is in the driver's seat: "
        f"Age: {row['Review_age_0']}. Education: {education_map.get(row['Review_education_0'], 'No Answer')}. "
        f"Gender: {row['Review_gender_0']}. Income: {income_map.get(row['Review_income_0'], 'No Answer')}. "
        f"Political (0 is Conservative, 1 is Progressive): {row['Review_political_0']}. "
        f"Religious (0 is Not Religious, 1 is Religious): {row['Review_religious_0']}. "
    )

df = pd.read_csv("data/MoralMachine/SharedResponsesSurveyUSA1M.csv")
if "Unnamed: 0" in df.columns: df = df.drop(columns=["Unnamed: 0"])
df['row_num'] = df.groupby('ResponseID').cumcount()
mm = df.pivot(index='ResponseID', columns='row_num')
mm.columns = [f"{col}_{num}" for col, num in mm.columns]
mm = mm.reset_index()
mm['Review_age_0'] = pd.to_numeric(mm['Review_age_0'], errors='coerce')
mm = mm[(mm['Review_age_0'] >= 0) & (mm['Review_age_0'] <= 120)]
_, mm_idx = train_test_split(mm.index, test_size=0.2, random_state=42)
mm_hold = mm.loc[mm_idx].sample(n=N, random_state=42)
mm_scenarios = [mm_prompt(r) for _, r in mm_hold.iterrows()]
mm_gts = [int(r['Saved_0']) for _, r in mm_hold.iterrows()]

print(f"\n{'='*50}")
print(f"  MoralMachine (n={N}) - LC-trained GRPO transfer")
print(f"{'='*50}")
for label, pipe in [("baseline", base_pipe), ("grpo-LC", lora_pipe)]:
    df = evaluate(pipe, label, mm_scenarios, mm_gts)
    print_results(df, label, "MoralMachine")

print("\nDone!")
PYEOF
