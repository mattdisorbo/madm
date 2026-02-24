import os, re, datetime, threading, random, time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
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

N_SAMPLES_BASE    = 10
N_SAMPLES_RF      = 10
N_SAMPLES_AUDITOR = 10
N_OAI        = 0
N_NANO       = 0
N_QWEN       = 2
N_QWEN_MED   = 2
N_QWEN_LARGE = 2
N_QWEN_XL    = 2
N_GLM        = 0
N_DEEPSEEK   = 0

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/hotel_bookings.csv")

# --- Load and clean data ---
print("Loading Hotel Bookings data...", flush=True)
data = pd.read_csv(DATA_PATH)
data = data.dropna(subset=[
    'is_canceled', 'arrival_date_year', 'arrival_date_month',
    'arrival_date_week_number', 'arrival_date_day_of_month',
    'stays_in_weekend_nights', 'stays_in_week_nights',
    'adults', 'is_repeated_guest', 'previous_cancellations',
    'required_car_parking_spaces', 'total_of_special_requests',
])
data['children'] = data['children'].fillna(0).astype(int)
data['babies'] = data['babies'].fillna(0).astype(int)
data = data.reset_index(drop=True)

print(f"Loaded {len(data)} bookings. Cancellation rate: {data['is_canceled'].mean():.3f}", flush=True)

# --- Train Random Forest ---
print("Training Random Forest...", flush=True)
features = [
    'arrival_date_year', 'arrival_date_month', 'arrival_date_week_number',
    'arrival_date_day_of_month', 'stays_in_weekend_nights', 'stays_in_week_nights',
    'adults', 'children', 'babies', 'is_repeated_guest', 'previous_cancellations',
    'required_car_parking_spaces', 'total_of_special_requests',
]
target = 'is_canceled'

df_clean = data[features + [target]].copy()
train_idx, holdout_idx = train_test_split(df_clean.index, test_size=0.2, random_state=42)

X_train = pd.get_dummies(df_clean.loc[train_idx, features], drop_first=True)
y_train = df_clean.loc[train_idx, target]
X_holdout = pd.get_dummies(df_clean.loc[holdout_idx, features], drop_first=True)
X_holdout = X_holdout.reindex(columns=X_train.columns, fill_value=0)

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
            out = local_pipes[model]([{"role": "user", "content": prompt}], max_new_tokens=2048)
        result = out[0]["generated_text"][-1]["content"]
    elif model == DEEPSEEK_MODEL:
        r = deepseek_client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}])
        result = r.choices[0].message.content.strip()
    else:
        r = oai_client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}])
        result = r.choices[0].message.content.strip()
    print(f"[{model.split(chr(47))[-1]}] {time.time()-t0:.1f}s", flush=True)
    return result

def create_prompt_base(row):
    party = f"{int(row['adults'])} adult(s)"
    if int(row['children']) > 0:
        party += f", {int(row['children'])} child(ren)"
    if int(row['babies']) > 0:
        party += f", {int(row['babies'])} baby/babies"

    repeated = "a repeated guest" if row['is_repeated_guest'] == 1 else "not a repeated guest"
    prev_cancel = int(row['previous_cancellations'])
    parking = int(row['required_car_parking_spaces'])
    special = int(row['total_of_special_requests'])

    return (
        f"You are helping predict Person 1's hotel booking decision. "
        f"Person 1 has booked a hotel stay arriving on {row['arrival_date_month']} {int(row['arrival_date_day_of_month'])}, "
        f"{int(row['arrival_date_year'])} (week {int(row['arrival_date_week_number'])}), "
        f"with {int(row['stays_in_weekend_nights'])} weekend night(s) and {int(row['stays_in_week_nights'])} weekday night(s). "
        f"The party consists of {party}. "
        f"Person 1 is {repeated} and has {prev_cancel} previous cancellation(s). "
        f"They have requested {parking} car parking space(s) and made {special} special request(s)."
    )

def get_llm_base(base_prompt, model):
    instructions = (
        "\n\nRespond with two numbers ONLY on two lines. "
        "First, respond 1 (cancel) or 0 (not cancel) based on your prediction of Person 1's true decision.\n"
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
    rf_label = "cancel" if rf_pred == 1 else "not cancel"
    instructions = (
        f"A random forest trained on a similar dataset predicts Person 1 would {rf_label} this booking, "
        f"based on a predicted probability of {rf_prob:.3f} that Person 1 would cancel. "
        "\n\nRespond with two numbers ONLY on two lines. "
        "First, respond 1 (cancel) or 0 (not cancel) based on your prediction of Person 1's true decision.\n"
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
            "Then conclude with exactly 'PREDICTION: 1' (cancel) or 'PREDICTION: 0' (not cancel).",
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
            "A decision is correct if it equals Person 1's true decision."
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
    human_response = int(row['is_canceled'])

    common = {
        'id': row_idx,
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
                 f"[CRITIQUE]\n{result['critique']}\n\n[DECISION PROMPT]\n{result['decision_prompt']}\n\n[DECISION]\n{result['decision']}")
        return {**common, 'llm_prediction': result['pred'], 'llm_delegate': result['del'],
                'llm_full_thought': result['full_thought'], 'llm_critique': result['critique'], 'trace': trace}

# --- Output ---
local_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../results/HotelBookings")
os.makedirs(local_dir, exist_ok=True)

def get_path(method, model):
    return os.path.join(local_dir, f'{method}_{model.split("/")[-1]}.csv')

df_existing = {}
for model, n in [(OAI_MODEL, N_OAI), (OAI_MODEL_NANO, N_NANO), (QWEN_MODEL, N_QWEN), (QWEN_MODEL_MED, N_QWEN_MED), (QWEN_MODEL_LARGE, N_QWEN_LARGE), (QWEN_MODEL_XL, N_QWEN_XL), (GLM_MODEL, N_GLM), (DEEPSEEK_MODEL, N_DEEPSEEK)]:
    if n > 0:
        for method in ["base", "rf", "auditor"]:
            path = get_path(method, model)
            try:
                df_existing[(method, model)] = pd.read_csv(path)
            except FileNotFoundError:
                df_existing[(method, model)] = pd.DataFrame()

results = []
completed = 0
total = (N_OAI + N_NANO + N_QWEN + N_QWEN_MED + N_QWEN_LARGE + N_QWEN_XL + N_GLM + N_DEEPSEEK) * (N_SAMPLES_BASE + N_SAMPLES_RF + N_SAMPLES_AUDITOR)
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
        for method, n_samples in [("base", N_SAMPLES_BASE), ("rf", N_SAMPLES_RF), ("auditor", N_SAMPLES_AUDITOR)]:
            if n_samples > 0:
                sampled = random.sample(holdout_indices, n * n_samples)
                for idx in sampled:
                    jobs.append((idx, method, model))

print(f"Starting {total} jobs | OAI {N_OAI}x(b={N_SAMPLES_BASE}, r={N_SAMPLES_RF}, a={N_SAMPLES_AUDITOR}) | Nano {N_NANO}x(b={N_SAMPLES_BASE}, r={N_SAMPLES_RF}, a={N_SAMPLES_AUDITOR}) | Qwen {N_QWEN}x | QwenMed {N_QWEN_MED}x | QwenLarge {N_QWEN_LARGE}x | QwenXL {N_QWEN_XL}x | GLM {N_GLM}x(b={N_SAMPLES_BASE}, r={N_SAMPLES_RF}, a={N_SAMPLES_AUDITOR}) | DeepSeek {N_DEEPSEEK}x(b={N_SAMPLES_BASE}, r={N_SAMPLES_RF}, a={N_SAMPLES_AUDITOR})", flush=True)
with ThreadPoolExecutor(max_workers=5) as executor:
    futures = [executor.submit(call_llm_tracked, idx, method, model) for idx, method, model in jobs]
    for f in as_completed(futures):
        f.result()

df_new = pd.DataFrame([r for r in results if r is not None])
for (method, model), group in df_new.groupby(['method', 'model']):
    path = get_path(method, model)
    print(f"Saved to {path}", flush=True)
    print(pd.read_csv(path)[['id', 'llm_prediction', 'human_response', 'llm_delegate', 'method', 'model']].to_string(), flush=True)
