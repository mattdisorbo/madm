import os, re, sys, datetime, threading, random
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from concurrent.futures import ThreadPoolExecutor, as_completed

QWEN_MODEL     = "Qwen/Qwen3.5-35B-A3B"
if len(sys.argv) > 1:
    QWEN_MODEL = sys.argv[1]

N_SAMPLES_BASE    = 50
N_SAMPLES_GLM = 50
N_SAMPLES_ADVERSARIAL = 50
N_OAI  = 0
N_NANO = 0
N_QWEN = 1

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/uber_bookings.csv")

# --- Load data ---
print("Loading Uber bookings data...", flush=True)
df = pd.read_csv(DATA_PATH)

# Filter to completed or cancelled rides only
df = df[df['Booking_Status'].isin(['Success', 'Canceled by Driver', 'Canceled by Customer'])].copy()
df['accepted'] = (df['Booking_Status'] == 'Success').astype(int)

# Clean location data
df['Pickup_Location'] = df['Pickup_Location'].fillna('Unknown').str.strip()
df['Drop_Location'] = df['Drop_Location'].fillna('Unknown').str.strip()
df = df.dropna(subset=['Pickup_Location', 'Drop_Location']).reset_index(drop=True)

print(f"Loaded {len(df)} rides. Acceptance rate: {df['accepted'].mean():.3f}", flush=True)

# --- Train GLM (Logistic Regression) ---
print("Training GLM...", flush=True)

le_pickup = LabelEncoder()
le_drop = LabelEncoder()
df['pickup_encoded'] = le_pickup.fit_transform(df['Pickup_Location'])
df['drop_encoded'] = le_drop.fit_transform(df['Drop_Location'])

features = ['pickup_encoded', 'drop_encoded']
target = 'accepted'

df_clean = df[features + [target, 'Pickup_Location', 'Drop_Location']].copy()
train_idx, holdout_idx = train_test_split(df_clean.index, test_size=0.2, random_state=42)

X_train = df_clean.loc[train_idx, features]
y_train = df_clean.loc[train_idx, target]
X_holdout = df_clean.loc[holdout_idx, features]

glm_model = LogisticRegression(max_iter=1000, random_state=42)
glm_model.fit(X_train, y_train)

glm_preds = glm_model.predict(X_holdout)
glm_probs = glm_model.predict_proba(X_holdout)[:, 1]
glm_acc = (glm_preds == df_clean.loc[holdout_idx, target].values).mean()
print(f"GLM accuracy: {glm_acc:.3f}", flush=True)

df_clean['glm_pred'] = np.nan
df_clean['glm_prob'] = np.nan
df_clean.loc[holdout_idx, 'glm_pred'] = glm_preds
df_clean.loc[holdout_idx, 'glm_prob'] = glm_probs

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
        import openai
        r = openai.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}])
        return r.choices[0].message.content.strip()

def create_prompt_base(row):
    return (
        f"You are helping Person 1 with a ride decision. "
        f"Person 1 has received a ride request going from {row['Pickup_Location']} to {row['Drop_Location']}."
    )

def get_llm_base(base_prompt, model):
    instructions = (
        "\n\nRespond with two numbers ONLY on two lines. "
        "First, respond 1 (accept ride) or 0 (decline ride) based on your prediction of Person 1's true decision.\n"
        "Second, respond 0 to implement your prediction of Person 1's true decision, "
        "or 1 to escalate to Person 1 so they can implement their true decision.\n"
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
    glm_label = "accept" if glm_pred == 1 else "decline"
    instructions = (
        f"A logistic regression model trained on a similar dataset predicts Person 1 would {glm_label} this ride, "
        f"based on a predicted probability of {glm_prob:.3f} that Person 1 would accept it. "
        "\n\nRespond with two numbers ONLY on two lines. "
        "First, respond 1 (accept ride) or 0 (decline ride) based on your prediction of Person 1's true decision.\n"
        "Second, respond 0 to implement your prediction of Person 1's true decision, "
        "or 1 to escalate to Person 1 so they can implement their true decision.\n"
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
            "Then conclude with exactly 'PREDICTION: 1' (accept) or 'PREDICTION: 0' (decline).",
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
            "Output 0 to implement your prediction, or 1 to escalate to Person 1.\n"
            "A decision is correct if it equals Person 1's true decision."
        )
        decision = llm(decision_prompt, model)
        del_match = re.search(r'[01]', decision.strip())
        final_del = int(del_match.group()) if del_match else 1
        return {"full_thought": thought, "pred": pred, "critique": critique, "decision_prompt": decision_prompt, "decision": decision, "del": final_del}
    except Exception as e:
        return {"full_thought": str(e), "pred": None, "critique": None, "decision": None, "del": None}

def call_llm(row_idx, method, model):
    row = df_clean.loc[row_idx]
    base = create_prompt_base(row)
    human_response = int(row[target])

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
        return {**common, 'llm_prediction': result['pred'], 'llm_escalate': result['del'], 'trace': trace}
    elif method == "glm":
        glm_pred_val = row['glm_pred']
        glm_prob_val = row['glm_prob']
        result = get_llm_glm(base, glm_pred_val, glm_prob_val, model)
        trace = f"[PROMPT]\n{result['full_prompt']}\n\n[RESPONSE]\n{result['response']}"
        return {**common, 'llm_prediction': result['pred'], 'llm_escalate': result['del'],
                'glm_pred': glm_pred_val, 'glm_prob': glm_prob_val, 'trace': trace}
    elif method == "adversarial":
        result = get_sequential_inference(base, model)
        trace = (f"[PROMPT]\n{base}\n\n[THOUGHT]\n{result['full_thought']}\n\n"
                 f"[CRITIQUE]\n{result['critique']}\n\n[DECISION PROMPT]\n{result['decision_prompt']}\n\n[DECISION]\n{result['decision']}")
        return {**common, 'llm_prediction': result['pred'], 'llm_escalate': result['del'],
                'llm_full_thought': result['full_thought'], 'llm_critique': result['critique'], 'trace': trace}

# --- Output ---
local_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../results/Uber")
os.makedirs(local_dir, exist_ok=True)

def get_path(method, model):
    return os.path.join(local_dir, f'{method}_{model.split("/")[-1]}.csv')

df_existing = {}
for model, n in [(QWEN_MODEL, N_QWEN)]:
    if n > 0:
        for method in ["base", "glm", "adversarial"]:
            path = get_path(method, model)
            try:
                df_existing[(method, model)] = pd.read_csv(path)
            except FileNotFoundError:
                df_existing[(method, model)] = pd.DataFrame()

results = []
completed = 0
total = N_QWEN * (N_SAMPLES_BASE + N_SAMPLES_GLM + N_SAMPLES_ADVERSARIAL)
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
for model, n in [(QWEN_MODEL, N_QWEN)]:
    if n > 0:
        for method, n_samples in [("base", N_SAMPLES_BASE), ("glm", N_SAMPLES_GLM), ("adversarial", N_SAMPLES_ADVERSARIAL)]:
            if n_samples > 0:
                sampled = random.sample(holdout_indices, n * n_samples)
                for idx in sampled:
                    jobs.append((idx, method, model))

print(f"Starting {total} jobs | Qwen {N_QWEN}x(b={N_SAMPLES_BASE}, g={N_SAMPLES_GLM}, a={N_SAMPLES_ADVERSARIAL})", flush=True)
with ThreadPoolExecutor(max_workers=1) as executor:
    futures = [executor.submit(call_llm_tracked, idx, method, model) for idx, method, model in jobs]
    for f in as_completed(futures):
        f.result()

df_new = pd.DataFrame([r for r in results if r is not None])
for (method, model), group in df_new.groupby(['method', 'model']):
    path = get_path(method, model)
    print(f"Saved to {path}", flush=True)
    print(pd.read_csv(path)[['id', 'pickup', 'drop', 'llm_prediction', 'human_response', 'llm_escalate', 'method', 'model']].to_string(), flush=True)
