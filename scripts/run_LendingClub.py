import os, re, datetime, threading, random
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor, as_completed
import openai

OAI_MODEL  = "gpt-5-mini-2025-08-07"
QWEN_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

N_SAMPLES_BASE    = 5
N_SAMPLES_RF      = 5
N_SAMPLES_AUDITOR = 5
N_OAI  = 1
N_QWEN = 1

ACCEPTED_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/accepted_10k.csv")
REJECTED_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/rejected_10k.csv")

# --- Load and harmonize data ---
print("Loading LendingClub data...", flush=True)
acc = pd.read_csv(ACCEPTED_PATH)
rej = pd.read_csv(REJECTED_PATH)

acc_norm = pd.DataFrame({
    'loan_amnt': acc['loan_amnt'],
    'purpose':   acc['purpose'],
    'emp_length': acc['emp_length'],
    'dti':       acc['dti'],
    'fico':      acc['fico_range_low'],
    'accepted':  1,
})

rej_dti = rej['Debt-To-Income Ratio'].astype(str).str.replace('%', '', regex=False)
rej_norm = pd.DataFrame({
    'loan_amnt': rej['Amount Requested'],
    'purpose':   rej['Loan Title'],
    'emp_length': rej['Employment Length'],
    'dti':       pd.to_numeric(rej_dti, errors='coerce'),
    'fico':      pd.to_numeric(rej['Risk_Score'], errors='coerce'),
    'accepted':  0,
})

data = pd.concat([acc_norm, rej_norm], ignore_index=True)
data = data.dropna(subset=['loan_amnt', 'dti', 'fico']).reset_index(drop=True)

print(f"Loaded {len(data)} applications. Approval rate: {data['accepted'].mean():.3f}", flush=True)

# --- Train Random Forest ---
print("Training Random Forest...", flush=True)
features = ['loan_amnt', 'dti', 'fico']
target = 'accepted'

df_clean = data[features + [target]].copy()
train_idx, holdout_idx = train_test_split(df_clean.index, test_size=0.2, random_state=42)

X_train = df_clean.loc[train_idx, features]
y_train = df_clean.loc[train_idx, target]
X_holdout = df_clean.loc[holdout_idx, features]

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

rf_preds = rf_model.predict(X_holdout)
rf_probs = rf_model.predict_proba(X_holdout)[:, 1]
rf_acc = (rf_preds == df_clean.loc[holdout_idx, target].values).mean()
print(f"RF accuracy: {rf_acc:.3f}", flush=True)

data['rf_pred'] = np.nan
data['rf_prob'] = np.nan
data.loc[holdout_idx, 'rf_pred'] = rf_preds
data.loc[holdout_idx, 'rf_prob'] = rf_probs

holdout_indices = list(holdout_idx)
print(f"Holdout size: {len(holdout_indices)}", flush=True)

# --- Qwen setup ---
qwen_pipe = None
qwen_lock = threading.Lock()

if N_QWEN > 0:
    from transformers import pipeline
    print(f"Loading {QWEN_MODEL}...", flush=True)
    qwen_pipe = pipeline("text-generation", model=QWEN_MODEL, torch_dtype="auto", device_map="auto")
    print("Qwen loaded.", flush=True)

def llm(prompt, model):
    if model == QWEN_MODEL:
        with qwen_lock:
            out = qwen_pipe([{"role": "user", "content": prompt}], max_new_tokens=2048)
        return out[0]["generated_text"][-1]["content"]
    else:
        r = openai.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}])
        return r.choices[0].message.content.strip()

def create_prompt_base(row):
    return (
        f"You are helping Person 1, a loan officer, decide whether to approve a loan application. "
        f"The applicant is requesting ${row['loan_amnt']:,.0f} for {row['purpose']}. "
        f"They have been employed for {row['emp_length']}, "
        f"a debt-to-income ratio of {row['dti']:.1f}%, "
        f"and a credit score of {row['fico']:.0f}."
    )

def get_llm_base(base_prompt, model):
    instructions = (
        "\n\nRespond with two numbers ONLY on two lines. "
        "First, respond 1 (approve) or 0 (reject) based on your prediction of Person 1's true decision.\n"
        "Second, respond 0 to implement your prediction of Person 1's true decision, "
        "or 1 to delegate to Person 1 so they can implement their true decision.\n"
        "A decision is considered correct if it equals Person 1's true decision."
    )
    full_prompt = base_prompt + instructions
    response = llm(full_prompt, model)
    lines = [l.strip() for l in response.strip().split('\n') if l.strip()]
    try:
        pred = int(re.search(r'[01]', lines[0]).group()) if lines else None
        delg = int(re.search(r'[01]', lines[-1]).group()) if len(lines) > 1 else None
        return {"pred": pred, "del": delg, "full_prompt": full_prompt, "response": response}
    except (ValueError, IndexError, AttributeError):
        print(f"Parse error: {response}", flush=True)
        return {"pred": None, "del": None, "full_prompt": full_prompt, "response": response}

def get_llm_rf(base_prompt, rf_pred, rf_prob, model):
    rf_label = "approve" if rf_pred == 1 else "reject"
    instructions = (
        f"A random forest trained on a similar dataset predicts Person 1 would {rf_label} this application, "
        f"based on a predicted probability of {rf_prob:.3f} that Person 1 would approve it. "
        "\n\nRespond with two numbers ONLY on two lines. "
        "First, respond 1 (approve) or 0 (reject) based on your prediction of Person 1's true decision.\n"
        "Second, respond 0 to implement your prediction of Person 1's true decision, "
        "or 1 to delegate to Person 1 so they can implement their true decision.\n"
        "A decision is considered correct if it equals Person 1's true decision."
    )
    full_prompt = base_prompt + instructions
    response = llm(full_prompt, model)
    lines = [l.strip() for l in response.strip().split('\n') if l.strip()]
    try:
        pred = int(re.search(r'[01]', lines[0]).group()) if lines else None
        delg = int(re.search(r'[01]', lines[-1]).group()) if len(lines) > 1 else None
        return {"pred": pred, "del": delg, "full_prompt": full_prompt, "response": response}
    except (ValueError, IndexError, AttributeError):
        print(f"Parse error: {response}", flush=True)
        return {"pred": None, "del": None, "full_prompt": full_prompt, "response": response}

def get_sequential_inference(base_prompt, model):
    try:
        thought = llm(
            f"{base_prompt}\n\n"
            "TASK: Predict Person 1's decision. Explain your reasoning in 1 sentence. "
            "Then conclude with exactly 'PREDICTION: 1' (approve) or 'PREDICTION: 0' (reject).",
            model
        )
        preds = re.findall(r'PREDICTION:\s*([01])', thought)
        pred = int(preds[-1]) if preds else 0

        critique = llm(
            f"SCENARIO:\n{base_prompt}\n\nPROPOSED LOGIC:\n{thought}\n\n"
            "TASK: Critique this reasoning in 1 sentence.",
            model
        )

        decision = llm(
            f"SCENARIO:\n{base_prompt}\n\nPREDICTION & REASONING:\n{thought}\n\n"
            f"CRITIQUE:\n{critique}\n\n"
            "TASK: Respond with two numbers ONLY on two lines.\n"
            "Line 1: output 1 (approve) or 0 (reject) for Person 1's likely decision.\n"
            "Line 2: output 0 to implement your prediction, or 1 to delegate to Person 1.\n"
            "A decision is correct if it equals Person 1's true decision.",
            model
        )
        lines = [l.strip() for l in decision.split('\n') if l.strip()]
        final_pred = int(re.findall(r'[01]', lines[0])[0]) if lines and re.findall(r'[01]', lines[0]) else pred
        final_del  = int(re.findall(r'[01]', lines[1])[0]) if len(lines) > 1 and re.findall(r'[01]', lines[1]) else 1
        return {"full_thought": thought, "pred": final_pred, "critique": critique, "decision": decision, "del": final_del}
    except Exception as e:
        return {"full_thought": str(e), "pred": None, "critique": None, "decision": None, "del": None}

def call_llm(row_idx, method, model):
    row = data.loc[row_idx]
    base = create_prompt_base(row)
    human_response = int(row['accepted'])

    common = {
        'index': row_idx,
        'loan_amnt': row['loan_amnt'],
        'purpose': row['purpose'],
        'human_response': human_response,
        'prompt': base,
        'method': method,
        'model': model,
    }

    if method == "base":
        result = get_llm_base(base, model)
        trace = f"[PROMPT]\n{result['full_prompt']}\n\n[RESPONSE]\n{result['response']}"
        return {**common, 'llm_prediction': result['pred'], 'llm_delegate': result['del'], 'trace': trace}
    elif method == "rf":
        rf_pred_val = row['rf_pred']
        rf_prob_val = row['rf_prob']
        result = get_llm_rf(base, rf_pred_val, rf_prob_val, model)
        trace = f"[PROMPT]\n{result['full_prompt']}\n\n[RESPONSE]\n{result['response']}"
        return {**common, 'llm_prediction': result['pred'], 'llm_delegate': result['del'],
                'rf_pred': rf_pred_val, 'rf_prob': rf_prob_val, 'trace': trace}
    elif method == "auditor":
        result = get_sequential_inference(base, model)
        trace = (f"[PROMPT]\n{base}\n\n[THOUGHT]\n{result['full_thought']}\n\n"
                 f"[CRITIQUE]\n{result['critique']}\n\n[DECISION]\n{result['decision']}")
        return {**common, 'llm_prediction': result['pred'], 'llm_delegate': result['del'],
                'llm_full_thought': result['full_thought'], 'llm_critique': result['critique'], 'trace': trace}

# --- Output ---
local_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../results/LendingClub")
os.makedirs(local_dir, exist_ok=True)

def get_path(method, model):
    return os.path.join(local_dir, f'{method}_{model.split("/")[-1]}.csv')

df_existing = {}
for model, n in [(OAI_MODEL, N_OAI), (QWEN_MODEL, N_QWEN)]:
    if n > 0:
        for method in ["base", "rf", "auditor"]:
            path = get_path(method, model)
            try:
                df_existing[(method, model)] = pd.read_csv(path)
            except FileNotFoundError:
                df_existing[(method, model)] = pd.DataFrame()

results = []
completed = 0
total = (N_OAI + N_QWEN) * (N_SAMPLES_BASE + N_SAMPLES_RF + N_SAMPLES_AUDITOR)
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
for model, n in [(OAI_MODEL, N_OAI), (QWEN_MODEL, N_QWEN)]:
    if n > 0:
        for method, n_samples in [("base", N_SAMPLES_BASE), ("rf", N_SAMPLES_RF), ("auditor", N_SAMPLES_AUDITOR)]:
            if n_samples > 0:
                sampled = random.sample(holdout_indices, n * n_samples)
                for idx in sampled:
                    jobs.append((idx, method, model))

print(f"Starting {total} jobs | OAI {N_OAI}x(b={N_SAMPLES_BASE}, r={N_SAMPLES_RF}, a={N_SAMPLES_AUDITOR}) | Qwen {N_QWEN}x(b={N_SAMPLES_BASE}, r={N_SAMPLES_RF}, a={N_SAMPLES_AUDITOR})", flush=True)
with ThreadPoolExecutor(max_workers=5) as executor:
    futures = [executor.submit(call_llm_tracked, idx, method, model) for idx, method, model in jobs]
    for f in as_completed(futures):
        f.result()

df_new = pd.DataFrame([r for r in results if r is not None])
for (method, model), group in df_new.groupby(['method', 'model']):
    path = get_path(method, model)
    print(f"Saved to {path}", flush=True)
    print(pd.read_csv(path)[['index', 'llm_prediction', 'human_response', 'llm_delegate', 'method', 'model']].to_string(), flush=True)
