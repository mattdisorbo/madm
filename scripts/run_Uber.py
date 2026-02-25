import os, re, datetime, threading, random, time
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor, as_completed
import openai

oai_client = openai.OpenAI()
deepseek_client = openai.OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
    base_url="https://api.deepseek.com",
)

OAI_MODEL        = "gpt-5-mini-2025-08-07"
OAI_MODEL_NANO   = "gpt-5-nano-2025-08-07"
QWEN_MODEL       = "Qwen/Qwen3-1.7B"
QWEN_MODEL_MED   = "Qwen/Qwen3-4B"
QWEN_MODEL_LARGE = "Qwen/Qwen3-8B"
QWEN_MODEL_XL    = "Qwen/Qwen3-14B"
GLM_MODEL        = "THUDM/glm-4-9b-chat-hf"
DEEPSEEK_MODEL   = "deepseek-chat"

N_SAMPLES_BASE    = 0
N_SAMPLES_GLM     = 0
N_SAMPLES_AUDITOR = 0
N_SAMPLES_COT     = 150
N_OAI        = 0
N_NANO       = 1
N_QWEN       = 0
N_QWEN_MED   = 0
N_QWEN_LARGE = 0
N_QWEN_XL    = 0
N_GLM        = 0
N_DEEPSEEK   = 0

import argparse as _ap
_p = _ap.ArgumentParser()
_p.add_argument('--model', type=str, default=None)
_p.add_argument('--n', type=int, default=None)
_args, _ = _p.parse_known_args()
if _args.model is not None and _args.n is not None:
    N_OAI        = 1 if OAI_MODEL        == _args.model else 0
    N_NANO       = 1 if OAI_MODEL_NANO   == _args.model else 0
    N_QWEN       = 1 if QWEN_MODEL       == _args.model else 0
    N_QWEN_MED   = 1 if QWEN_MODEL_MED   == _args.model else 0
    N_QWEN_LARGE = 1 if QWEN_MODEL_LARGE == _args.model else 0
    N_QWEN_XL    = 1 if QWEN_MODEL_XL    == _args.model else 0
    N_GLM        = 1 if GLM_MODEL        == _args.model else 0
    N_DEEPSEEK   = 1 if DEEPSEEK_MODEL   == _args.model else 0
    N_SAMPLES_BASE    = _args.n
    N_SAMPLES_AUDITOR = _args.n
    N_SAMPLES_RF      = _args.n
    N_SAMPLES_OLS     = _args.n
    N_SAMPLES_GLM     = _args.n
    N_SAMPLES_COT     = _args.n

# Booking_Status values kept: "Success" (label 0) and "Canceled by Driver" (label 1)
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/Bookings.csv")

# --- Load and clean data ---
print("Loading Uber data...", flush=True)
data = pd.read_csv(DATA_PATH)
data = data[data["Booking_Status"].isin(["Success", "Canceled by Driver"])].copy()
data["cancelled"] = (data["Booking_Status"] == "Canceled by Driver").astype(int)
data = data.dropna(subset=["Pickup_Location", "Drop_Location"])
data = data.reset_index(drop=True)

print(f"Loaded {len(data)} rides. Driver cancel rate: {data['cancelled'].mean():.3f}", flush=True)

# --- Train Logistic Regression ---
print("Training Logistic Regression...", flush=True)
features_df = pd.get_dummies(data[['Pickup_Location', 'Drop_Location']])
target = data['cancelled']

train_idx, holdout_idx = train_test_split(data.index, test_size=0.2, random_state=42)

X_train = features_df.loc[train_idx]
y_train = target.loc[train_idx]
X_holdout = features_df.loc[holdout_idx]

glm_model = LogisticRegression(max_iter=1000, random_state=42)
glm_model.fit(X_train, y_train)

glm_preds = glm_model.predict(X_holdout)
glm_probs = glm_model.predict_proba(X_holdout)[:, 1]
glm_acc = (glm_preds == target.loc[holdout_idx].values).mean()
print(f"Logistic Regression accuracy: {glm_acc:.3f}", flush=True)

data['glm_pred'] = np.nan
data['glm_prob'] = np.nan
data.loc[holdout_idx, 'glm_pred'] = glm_preds
data.loc[holdout_idx, 'glm_prob'] = glm_probs

holdout_indices = list(holdout_idx)
print(f"Holdout size: {len(holdout_indices)}", flush=True)

# --- Local model setup ---
local_pipes = {}
local_locks = {}

for _m, _n in [(QWEN_MODEL, N_QWEN), (QWEN_MODEL_MED, N_QWEN_MED), (QWEN_MODEL_LARGE, N_QWEN_LARGE), (QWEN_MODEL_XL, N_QWEN_XL), (GLM_MODEL, N_GLM)]:
    if _n > 0:
        from transformers import pipeline
        print(f"Loading {_m}...", flush=True)
        _t = time.time()
        local_pipes[_m] = pipeline("text-generation", model=_m, torch_dtype="auto", device_map="auto", trust_remote_code=True)
        print(f"Loaded {_m.split(chr(47))[-1]} in {time.time()-_t:.1f}s", flush=True)
        local_locks[_m] = threading.Lock()
        print(f"{_m} loaded.", flush=True)

def llm(prompt, model):
    t0 = time.time()
    if model in local_pipes:
        with local_locks[model]:
            out = local_pipes[model]([{"role": "system", "content": "/no_think"}, {"role": "user", "content": prompt}], max_new_tokens=128)
        result = out[0]["generated_text"][-1]["content"]
        result = re.sub(r'<think>.*?</think>', '', result, flags=re.DOTALL).strip()
    elif model == DEEPSEEK_MODEL:
        r = deepseek_client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}])
        result = r.choices[0].message.content.strip()
    else:
        r = oai_client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}])
        result = r.choices[0].message.content.strip()
    print(f"[{model.split(chr(47))[-1]}] {time.time()-t0:.1f}s", flush=True)
    return result

def create_prompt_base(row):
    return (
        f"You are helping Person 1 with a ride decision. "
        f"Person 1 has received a ride request going from {row['Pickup_Location']} to {row['Drop_Location']}."
    )

def get_llm_base(base_prompt, model):
    instructions = (
        "\n\nRespond with two numbers ONLY on two lines. "
        "First, respond 1 (cancel) or 0 (accept) based on your prediction of Person 1's true decision.\n"
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

def get_llm_glm(base_prompt, glm_pred, glm_prob, model):
    glm_label = "cancel" if glm_pred == 1 else "accept"
    instructions = (
        f"A logistic regression trained on a similar dataset predicts Person 1 would {glm_label} this ride, "
        f"based on a predicted probability of {glm_prob:.3f} that Person 1 would cancel it. "
        "\n\nRespond with two numbers ONLY on two lines. "
        "First, respond 1 (cancel) or 0 (accept) based on your prediction of Person 1's true decision.\n"
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

def get_llm_cot(base_prompt, model):
    instructions = (
        "\n\nRespond with two numbers ONLY on two lines. "
        "First, respond 1 (cancel) or 0 (accept) based on your prediction of Person 1's true decision.\n"
        "Second, respond 0 to implement your prediction of Person 1's true decision, "
        "or 1 to delegate to Person 1 so they can implement their true decision.\n"
        "A decision is considered correct if it equals Person 1's true decision."
    )
    full_prompt = base_prompt + instructions
    t0 = time.time()
    r = oai_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": full_prompt}],
        reasoning_effort="medium",
    )
    print(f"[{model.split(chr(47))[-1]} cot] {time.time()-t0:.1f}s", flush=True)
    response = r.choices[0].message.content.strip()
    reasoning_tokens = getattr(getattr(r.usage, 'completion_tokens_details', None), 'reasoning_tokens', None)
    lines = [l.strip() for l in response.strip().split('\n') if l.strip()]
    try:
        pred = int(re.search(r'[01]', lines[0]).group()) if lines else None
        delg = int(re.search(r'[01]', lines[-1]).group()) if len(lines) > 1 else None
    except (ValueError, IndexError, AttributeError):
        pred, delg = None, None
    return {"full_prompt": full_prompt, "response": response, "pred": pred, "del": delg, "reasoning_tokens": reasoning_tokens}

def get_sequential_inference(base_prompt, model):
    try:
        thought = llm(
            f"{base_prompt}\n\n"
            "TASK: Predict Person 1's decision. Explain your reasoning in 1 sentence. "
            "Then conclude with exactly 'PREDICTION: 1' (cancel) or 'PREDICTION: 0' (accept).",
            model
        )
        preds = re.findall(r'PREDICTION:\s*([01])', thought)
        pred = int(preds[-1]) if preds else 0

        critique = llm(
            f"SCENARIO:\n{base_prompt}\n\nPROPOSED LOGIC:\n{thought}\n\n"
            "TASK: Critique this reasoning in 1 sentence.",
            model
        )

        decision_prompt = (
            f"SCENARIO:\n{base_prompt}\n\nPREDICTION & REASONING:\n{thought}\n\n"
            f"CRITIQUE:\n{critique}\n\n"
            "TASK: Respond with a digit ONLY.\n"
            "Output 0 to implement your prediction, or 1 to delegate to Person 1.\n"
            "The ground truth is Person 1's true decision."
        )
        decision = llm(decision_prompt, model)
        del_match = re.search(r'[01]', decision.strip())
        final_del = int(del_match.group()) if del_match else 1
        return {"full_thought": thought, "pred": pred, "critique": critique, "decision_prompt": decision_prompt, "decision": decision, "del": final_del}
    except Exception as e:
        return {"full_thought": str(e), "pred": None, "critique": None, "decision": None, "del": None}

def call_llm(row_idx, method, model):
    row = data.loc[row_idx]
    base = create_prompt_base(row)
    human_response = int(row['cancelled'])

    common = {
        'id': row_idx,
        'pickup': row['Pickup_Location'],
        'drop': row['Drop_Location'],
        'human_response': human_response,
        'prompt': base,
        'method': method,
        'model': model,
    }

    if method == "base":
        result = get_llm_base(base, model)
        trace = f"[PROMPT]\n{result['full_prompt']}\n\n[RESPONSE]\n{result['response']}"
        return {**common, 'llm_prediction': result['pred'], 'llm_delegate': result['del'], 'trace': trace}
    elif method == "glm":
        glm_pred_val = row['glm_pred']
        glm_prob_val = row['glm_prob']
        result = get_llm_glm(base, glm_pred_val, glm_prob_val, model)
        trace = f"[PROMPT]\n{result['full_prompt']}\n\n[RESPONSE]\n{result['response']}"
        return {**common, 'llm_prediction': result['pred'], 'llm_delegate': result['del'],
                'glm_pred': glm_pred_val, 'glm_prob': glm_prob_val, 'trace': trace}
    elif method == "cot":
        result = get_llm_cot(base, model)
        trace = f"[PROMPT]\n{result['full_prompt']}\n\n[RESPONSE]\n{result['response']}"
        return {**common, 'llm_prediction': result['pred'], 'llm_delegate': result['del'], 'reasoning_tokens': result['reasoning_tokens'], 'trace': trace}
    elif method == "auditor":
        result = get_sequential_inference(base, model)
        trace = (f"[PROMPT]\n{base}\n\n[THOUGHT]\n{result['full_thought']}\n\n"
                 f"[CRITIQUE]\n{result['critique']}\n\n[DECISION PROMPT]\n{result['decision_prompt']}\n\n[DECISION]\n{result['decision']}")
        return {**common, 'llm_prediction': result['pred'], 'llm_delegate': result['del'],
                'llm_full_thought': result['full_thought'], 'llm_critique': result['critique'], 'trace': trace}

# --- Output ---
local_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../results/Uber")
os.makedirs(local_dir, exist_ok=True)

def get_path(method, model):
    return os.path.join(local_dir, f'{method}_{model.split("/")[-1]}.csv')

df_existing = {}
for model, n in [(OAI_MODEL, N_OAI), (OAI_MODEL_NANO, N_NANO), (QWEN_MODEL, N_QWEN), (QWEN_MODEL_MED, N_QWEN_MED), (QWEN_MODEL_LARGE, N_QWEN_LARGE), (QWEN_MODEL_XL, N_QWEN_XL), (GLM_MODEL, N_GLM), (DEEPSEEK_MODEL, N_DEEPSEEK)]:
    if n > 0:
        for method in ["base", "glm", "auditor", "cot"]:
            path = get_path(method, model)
            try:
                df_existing[(method, model)] = pd.read_csv(path)
            except FileNotFoundError:
                df_existing[(method, model)] = pd.DataFrame()

results = []
completed = 0
total = (N_OAI + N_NANO + N_QWEN + N_QWEN_MED + N_QWEN_LARGE + N_QWEN_XL + N_GLM + N_DEEPSEEK) * (N_SAMPLES_BASE + N_SAMPLES_GLM + N_SAMPLES_AUDITOR) + N_NANO * N_SAMPLES_COT
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
for model, n in [(OAI_MODEL, N_OAI), (OAI_MODEL_NANO, N_NANO), (QWEN_MODEL, N_QWEN), (QWEN_MODEL_MED, N_QWEN_MED), (QWEN_MODEL_LARGE, N_QWEN_LARGE), (QWEN_MODEL_XL, N_QWEN_XL), (GLM_MODEL, N_GLM), (DEEPSEEK_MODEL, N_DEEPSEEK)]:
    if n > 0:
        methods = [("base", N_SAMPLES_BASE), ("glm", N_SAMPLES_GLM), ("auditor", N_SAMPLES_AUDITOR)]
        if model == OAI_MODEL_NANO:
            methods.append(("cot", N_SAMPLES_COT))
        for method, n_samples in methods:
            if n_samples > 0:
                sampled = random.sample(holdout_indices, n * n_samples)
                for idx in sampled:
                    jobs.append((idx, method, model))

print(f"Starting {total} jobs | OAI {N_OAI}x(b={N_SAMPLES_BASE}, g={N_SAMPLES_GLM}, a={N_SAMPLES_AUDITOR}) | Nano {N_NANO}x(b={N_SAMPLES_BASE}, c={N_SAMPLES_COT}, g={N_SAMPLES_GLM}, a={N_SAMPLES_AUDITOR}) | Qwen {N_QWEN}x | QwenMed {N_QWEN_MED}x | QwenLarge {N_QWEN_LARGE}x | QwenXL {N_QWEN_XL}x | GLM {N_GLM}x | DeepSeek {N_DEEPSEEK}x", flush=True)
with ThreadPoolExecutor(max_workers=5) as executor:
    futures = [executor.submit(call_llm_tracked, idx, method, model) for idx, method, model in jobs]
    for f in as_completed(futures):
        f.result()

df_new = pd.DataFrame([r for r in results if r is not None])
for (method, model), group in df_new.groupby(['method', 'model']):
    path = get_path(method, model)
    print(f"Saved to {path}", flush=True)
    print(pd.read_csv(path)[['id', 'pickup', 'drop', 'llm_prediction', 'human_response', 'llm_delegate', 'method', 'model']].to_string(), flush=True)
