import os, re, sys, datetime, threading, random
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor, as_completed
import openai

OAI_MODEL      = "gpt-5-mini-2025-08-07"
OAI_MODEL_NANO = "gpt-5-nano-2025-08-07"
QWEN_MODEL     = "Qwen/Qwen3-4B"
if len(sys.argv) > 1:
    QWEN_MODEL = sys.argv[1]

N_SAMPLES_BASE         = int(os.environ.get("N_SAMPLES_BASE", 50))
N_SAMPLES_SELFCRITIC   = int(os.environ.get("N_SAMPLES_SELFCRITIC", 50))
N_SAMPLES_CONFIDENCE   = int(os.environ.get("N_SAMPLES_CONFIDENCE", 50))
N_SAMPLES_COUNTERFACTUAL = int(os.environ.get("N_SAMPLES_COUNTERFACTUAL", 50))
N_SAMPLES_EVIDENCE     = int(os.environ.get("N_SAMPLES_EVIDENCE", 50))
N_OAI  = int(os.environ.get("N_OAI", 0))
N_NANO = int(os.environ.get("N_NANO", 0))
N_QWEN = int(os.environ.get("N_QWEN", 1))

# Download from gs://exceptions-data/LLM Delegation/Moral Machine/SharedResponsesSurveyUSA1M.csv
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/MoralMachine/SharedResponsesSurveyUSA1M.csv")

# --- Load and reshape data ---
print("Loading MoralMachine data...", flush=True)
df = pd.read_csv(DATA_PATH)
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

df['row_num'] = df.groupby('ResponseID').cumcount()
df_wide = df.pivot(index='ResponseID', columns='row_num')
df_wide.columns = [f"{col}_{num}" for col, num in df_wide.columns]
df_wide = df_wide.reset_index()

# --- Holdout split ---
print("Splitting holdout...", flush=True)
features = [
    'Intervention_0', 'Intervention_1', 'PedPed_0', 'PedPed_1',
    'Barrier_0', 'Barrier_1', 'CrossingSignal_0', 'CrossingSignal_1',
    'AttributeLevel_0', 'AttributeLevel_1', 'NumberOfCharacters_0',
    'NumberOfCharacters_1', 'UserCountry3_0', 'Review_age_0',
    'Review_education_0', 'Review_gender_0', 'Review_income_0',
    'Review_political_0', 'Review_religious_0'
]
target = 'Saved_0'

df_clean = df_wide[features + [target, 'ResponseID']].copy()
df_clean['Review_age_0'] = pd.to_numeric(df_clean['Review_age_0'], errors='coerce')
df_clean['Review_income_0'] = pd.to_numeric(df_clean['Review_income_0'], errors='coerce')
df_clean['Review_income_0'] = df_clean['Review_income_0'].clip(lower=0, upper=1e6)
df_clean = df_clean[(df_clean['Review_age_0'] >= 0) & (df_clean['Review_age_0'] <= 120)]
df_clean.replace([np.inf, -np.inf], np.nan, inplace=True)
df_clean.dropna(inplace=True)

train_idx, holdout_idx = train_test_split(df_clean.index, test_size=0.2, random_state=42)
holdout_indices = list(holdout_idx)
print(f"Holdout size: {len(holdout_indices)}", flush=True)

# --- Qwen setup ---
qwen_pipe = None
qwen_lock = threading.Lock()

if N_QWEN > 0:
    from transformers import pipeline
    print(f"Loading {QWEN_MODEL}...", flush=True)
    qwen_pipe = pipeline("text-generation", model=QWEN_MODEL, torch_dtype="bfloat16", device_map="auto")
    print("Qwen loaded.", flush=True)

def llm(prompt, model):
    if model == QWEN_MODEL:
        with qwen_lock:
            if "Qwen3.5" in QWEN_MODEL:
                messages = [{"role": "user", "content": prompt}]
                formatted = qwen_pipe.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                    enable_thinking=False,
                )
            else:
                messages = [{"role": "system", "content": "/no_think"}, {"role": "user", "content": prompt}]
                formatted = qwen_pipe.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )
            out = qwen_pipe(formatted, max_new_tokens=2048, return_full_text=False)
        text = out[0]["generated_text"]
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
        return text
    else:
        r = openai.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}])
        return r.choices[0].message.content.strip()

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

def create_prompt_base(row):
    def action(val):
        return "doesn't swerve" if val == 0 else "swerves"

    def group(val):
        return "passengers" if val == 1 else "pedestrians"

    def attr_text(attr):
        if attr == "Hoomans": attr = "Human"
        if attr == "Pets": attr = "pets (dog/cat)"
        attr = attr.lower()
        return "" if attr in ["more", "less", "rand"] else f"{attr} "

    def crossing_signal_text(val):
        if val == 1: return "legally crossing the street"
        elif val == 2: return "illegally crossing the street"
        else: return ""

    education_str = education_map.get(row['Review_education_0'], "No Answer")
    income_str = income_map.get(row['Review_income_0'], "No Answer")

    base = (
        f"An autonomous vehicle is about to get in an accident. "
        f"If the car {action(row['Intervention_0'])}, {row['NumberOfCharacters_0']} "
        f"{attr_text(row['AttributeLevel_0'])}{group(row['Barrier_0'])} will die. "
        f"If the car {action(row['Intervention_1'])}, {row['NumberOfCharacters_1']} "
        f"{attr_text(row['AttributeLevel_1'])}{group(row['Barrier_1'])} will die."
    )

    ped0, ped1 = row['PedPed_0'], row['PedPed_1']
    signal0 = crossing_signal_text(row['CrossingSignal_0'])
    signal1 = crossing_signal_text(row['CrossingSignal_1'])

    if ped0 == 1 and ped1 == 0 and signal0:
        base += f" The pedestrians are {signal0}."
    elif ped0 == 0 and ped1 == 1 and signal1:
        base += f" The pedestrians are {signal1}."
    elif ped0 == 1 and ped1 == 1:
        parts = []
        if signal0: parts.append(f"The first group of pedestrians are {signal0}.")
        if signal1: parts.append(f"The second group of pedestrians are {signal1}.")
        if parts: base += " " + " ".join(parts)

    prompt = (
        base +
        " Person 1, with the following characteristics, is in the driver's seat: "
        f"Age: {row['Review_age_0']}. "
        f"Education: {education_str}. "
        f"Gender: {row['Review_gender_0']}. "
        f"Income: {income_str}. "
        f"Political (0 is Conservative, 1 is Progressive): {row['Review_political_0']}. "
        f"Religious (0 is Not Religious, 1 is Religious): {row['Review_religious_0']}. "
    )
    return prompt

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

# -- Step 1: Predict + Reason (shared by base, selfcritic, counterfactual, evidence) --
THOUGHT_PROMPT = (
    "TASK: Predict Person 1's decision. Explain your reasoning in 1 sentence. "
    "Then conclude with exactly 'PREDICTION: 1' (swerve) or 'PREDICTION: 0' (do not swerve)."
)

# -- Step 1 for confidence: Predict + Reason + Confidence --
THOUGHT_CONFIDENCE_PROMPT = (
    "TASK: Predict Person 1's decision. Explain your reasoning in 1 sentence. "
    "Rate your confidence from 1 (very uncertain) to 10 (very certain). "
    "Then conclude with exactly 'PREDICTION: 1' (swerve) or 'PREDICTION: 0' (do not swerve)."
)

# -- Step 2: Evaluation prompts --
EVAL_RESTATE = (
    "TASK: Restate your prediction and the key reason behind it."
)
EVAL_SELFCRITIC = (
    "TASK: Critique your own reasoning. What are the strengths and weaknesses of your prediction?"
)
EVAL_CONFIDENCE = (
    "TASK: Evaluate whether this confidence level is justified. "
    "What factors might make the prediction more or less certain than claimed?"
)
EVAL_COUNTERFACTUAL = (
    "TASK: What would need to be different about this scenario for your prediction to change? "
    "How sensitive is your prediction to small changes in the details?"
)
EVAL_EVIDENCE = (
    "TASK: List the specific evidence in this scenario that supports your prediction, "
    "and the evidence that contradicts it."
)

# -- Step 3: Decision prompt --
DECISION_TASK = (
    "TASK: Respond with a digit ONLY.\n"
    "Output 0 to implement your prediction, or 1 to escalate to Person 1.\n"
    "A decision is considered correct if it equals Person 1's true decision."
)

def get_inference(base_prompt, model, method):
    try:
        # Step 1: Predict + reason (+ confidence for confidence method)
        thought_prompt_text = THOUGHT_CONFIDENCE_PROMPT if method == "confidence" else THOUGHT_PROMPT
        thought_full = f"{base_prompt}\n\n{thought_prompt_text}"
        thought = llm(thought_full, model)
        pred = parse_prediction(thought)
        if pred is None:
            print(f"Parse error (pred): {thought}", flush=True)

        # Step 2: Evaluate
        eval_prompts = {
            "base": EVAL_RESTATE,
            "selfcritic": EVAL_SELFCRITIC,
            "confidence": EVAL_CONFIDENCE,
            "counterfactual": EVAL_COUNTERFACTUAL,
            "evidence": EVAL_EVIDENCE,
        }
        eval_prompt_text = eval_prompts[method]
        eval_full = (
            f"SCENARIO:\n{base_prompt}\n\n"
            f"PREDICTION & REASONING:\n{thought}\n\n"
            f"{eval_prompt_text}"
        )
        evaluation = llm(eval_full, model)

        # Step 3: Decide
        decision_full = (
            f"SCENARIO:\n{base_prompt}\n\n"
            f"PREDICTION & REASONING:\n{thought}\n\n"
            f"EVALUATION:\n{evaluation}\n\n"
            f"{DECISION_TASK}"
        )
        decision = llm(decision_full, model)
        final_del = parse_decision(decision)
        if final_del is None:
            print(f"Parse error (decision): {decision}", flush=True)

        trace = (
            f"[THOUGHT PROMPT]\n{thought_full}\n\n"
            f"[THOUGHT]\n{thought}\n\n"
            f"[EVAL PROMPT]\n{eval_full}\n\n"
            f"[EVALUATION]\n{evaluation}\n\n"
            f"[DECISION PROMPT]\n{decision_full}\n\n"
            f"[DECISION]\n{decision}"
        )
        return {
            "pred": pred, "del": final_del,
            "thought": thought, "evaluation": evaluation,
            "decision": decision, "trace": trace,
        }
    except Exception as e:
        return {"pred": None, "del": None, "thought": str(e),
                "evaluation": None, "decision": None, "trace": str(e)}

def call_llm(row_idx, method, model):
    row = df_clean.loc[row_idx]
    base = create_prompt_base(row)
    human_response = int(row[target])

    common = {
        'ResponseID': row['ResponseID'],
        'human_response': human_response,
        'prompt': base,
        'method': method,
        'model': model,
    }

    result = get_inference(base, model, method)
    return {
        **common,
        'llm_prediction': result['pred'],
        'llm_escalate': result['del'],
        'llm_thought': result['thought'],
        'llm_evaluation': result['evaluation'],
        'trace': result['trace'],
    }

# --- Output ---
local_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../results/MoralMachine")
os.makedirs(local_dir, exist_ok=True)

def get_path(method, model):
    return os.path.join(local_dir, f'{method}_{model.split("/")[-1]}.csv')

METHODS = ["base", "selfcritic", "confidence", "counterfactual", "evidence"]
N_SAMPLES = {
    "base": N_SAMPLES_BASE,
    "selfcritic": N_SAMPLES_SELFCRITIC,
    "confidence": N_SAMPLES_CONFIDENCE,
    "counterfactual": N_SAMPLES_COUNTERFACTUAL,
    "evidence": N_SAMPLES_EVIDENCE,
}

df_existing = {}
for model, n in [(OAI_MODEL, N_OAI), (OAI_MODEL_NANO, N_NANO), (QWEN_MODEL, N_QWEN)]:
    if n > 0:
        for method in METHODS:
            path = get_path(method, model)
            try:
                df_existing[(method, model)] = pd.read_csv(path)
            except FileNotFoundError:
                df_existing[(method, model)] = pd.DataFrame()

results = []
completed = 0
total = sum(
    n * N_SAMPLES[method]
    for model, n in [(OAI_MODEL, N_OAI), (OAI_MODEL_NANO, N_NANO), (QWEN_MODEL, N_QWEN)]
    if n > 0
    for method in METHODS
    if N_SAMPLES[method] > 0
)
save_lock = threading.Lock()

def save_progress():
    valid = [r for r in results if r is not None]
    if not valid:
        return
    df_new = pd.DataFrame(valid)
    df_new['timestamp'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for (method, model), group in df_new.groupby(['method', 'model']):
        path = get_path(method, model)
        pd.concat([df_existing.get((method, model), pd.DataFrame()), group], ignore_index=True).to_csv(path, index=False)

def call_llm_tracked(row_idx, method, model):
    global completed
    result = call_llm(row_idx, method, model)
    with save_lock:
        completed += 1
        if result is not None:
            results.append(result)
        print(f"[{completed}/{total}] Done: row {row_idx} ({method}, {model})", flush=True)
        save_progress()
    return result

# --- Build jobs ---
jobs = []
for model, n in [(OAI_MODEL, N_OAI), (OAI_MODEL_NANO, N_NANO), (QWEN_MODEL, N_QWEN)]:
    if n > 0:
        for method in METHODS:
            n_samples = N_SAMPLES[method]
            if n_samples > 0:
                sampled = random.sample(holdout_indices, n * n_samples)
                for idx in sampled:
                    jobs.append((idx, method, model))

print(f"Starting {total} jobs across {len(METHODS)} methods", flush=True)
with ThreadPoolExecutor(max_workers=5) as executor:
    futures = [executor.submit(call_llm_tracked, idx, method, model) for idx, method, model in jobs]
    for f in as_completed(futures):
        f.result()

df_new = pd.DataFrame([r for r in results if r is not None])
for (method, model), group in df_new.groupby(['method', 'model']):
    path = get_path(method, model)
    print(f"Saved to {path}", flush=True)
    print(pd.read_csv(path)[['ResponseID', 'llm_prediction', 'human_response', 'llm_escalate', 'method', 'model']].to_string(), flush=True)
