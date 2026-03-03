import os, re, sys, datetime, threading, random
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor, as_completed

QWEN_MODEL     = "Qwen/Qwen3.5-35B-A3B"
if len(sys.argv) > 1:
    QWEN_MODEL = sys.argv[1]

N_SAMPLES_BASE    = 50
N_SAMPLES_RF = 50
N_SAMPLES_ADVERSARIAL = 50
N_OAI  = int(os.environ.get("N_OAI", 0))
N_NANO = int(os.environ.get("N_NANO", 0))
N_QWEN = int(os.environ.get("N_QWEN", 1))

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/hotel_bookings.csv")

# --- Load data ---
print("Loading HotelBookings data...", flush=True)
df = pd.read_csv(DATA_PATH)

# Create arrival date string
df['arrival_date'] = pd.to_datetime(
    df['arrival_date_year'].astype(str) + '-' +
    df['arrival_date_month'] + '-' +
    df['arrival_date_day_of_month'].astype(str),
    format='%Y-%B-%d',
    errors='coerce'
)

# Target: is_canceled (0 = kept booking, 1 = cancelled)
# We predict from the guest's perspective: 1 = will keep, 0 = will cancel
df['kept_booking'] = 1 - df['is_canceled']

# Features for RF
features = [
    'arrival_date_week_number', 'stays_in_weekend_nights', 'stays_in_week_nights',
    'adults', 'children', 'is_repeated_guest', 'previous_cancellations',
    'required_car_parking_spaces', 'total_of_special_requests',
]
target = 'kept_booking'

df_clean = df[features + [target, 'arrival_date', 'arrival_date_month',
              'arrival_date_day_of_month', 'arrival_date_year']].copy()
df_clean = df_clean.dropna(subset=features).reset_index(drop=True)

print(f"Loaded {len(df_clean)} bookings. Keep rate: {df_clean[target].mean():.3f}", flush=True)

# --- Train Random Forest ---
print("Training Random Forest...", flush=True)
train_idx, holdout_idx = train_test_split(df_clean.index, test_size=0.2, random_state=42)

X_train = df_clean.loc[train_idx, features].fillna(0)
y_train = df_clean.loc[train_idx, target]
X_holdout = df_clean.loc[holdout_idx, features].fillna(0)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

rf_preds = rf_model.predict(X_holdout)
rf_probs = rf_model.predict_proba(X_holdout)[:, 1]
rf_acc = (rf_preds == df_clean.loc[holdout_idx, target].values).mean()
print(f"RF accuracy: {rf_acc:.3f}", flush=True)

df_clean['rf_pred'] = np.nan
df_clean['rf_prob'] = np.nan
df_clean.loc[holdout_idx, 'rf_pred'] = rf_preds
df_clean.loc[holdout_idx, 'rf_prob'] = rf_probs

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
    date_str = row['arrival_date'].strftime('%B %d, %Y') if pd.notna(row['arrival_date']) else f"{row['arrival_date_month']} {row['arrival_date_day_of_month']}, {row['arrival_date_year']}"
    week = int(row['arrival_date_week_number'])
    weekend = int(row['stays_in_weekend_nights'])
    weekday = int(row['stays_in_week_nights'])
    adults = int(row['adults'])
    children = int(row['children']) if pd.notna(row['children']) else 0
    repeated = "a repeated guest" if row['is_repeated_guest'] == 1 else "not a repeated guest"
    prev_cancel = int(row['previous_cancellations'])
    parking = int(row['required_car_parking_spaces'])
    special = int(row['total_of_special_requests'])

    prompt = (
        f"You are helping predict Person 1's hotel booking decision. "
        f"Person 1 has booked a hotel stay arriving on {date_str} (week {week}), "
        f"with {weekend} weekend night(s) and {weekday} weekday night(s). "
        f"The party consists of {adults} adult(s)"
    )
    if children > 0:
        prompt += f", {children} child(ren)"
    prompt += (
        f". Person 1 is {repeated} and has {prev_cancel} previous cancellation(s). "
        f"They have requested {parking} car parking space(s) and made {special} special request(s)."
    )
    return prompt

def get_llm_base(base_prompt, model):
    # Step 1: Get prediction
    pred_prompt = base_prompt + "\n\nRespond with a single digit ONLY. 1 (will keep booking) or 0 (will cancel) based on your prediction of Person 1's true decision."
    pred_response = llm(pred_prompt, model)
    try:
        pred = int(re.search(r'[01]', pred_response.strip()).group())
    except (ValueError, AttributeError):
        print(f"Parse error (pred): {pred_response}", flush=True)
        pred = None

    # Step 2: Get escalation decision
    esc_prompt = (
        f"{base_prompt}\n\nYour prediction: {pred_response.strip()}\n\n"
        "Respond with a single digit ONLY.\n"
        "Output 0 to implement your prediction of Person 1's true decision, "
        "or 1 to escalate to Person 1 so they can implement their true decision.\n"
        "A decision is considered correct if it equals Person 1's true decision."
    )
    esc_response = llm(esc_prompt, model)
    try:
        delg = int(re.search(r'[01]', esc_response.strip()).group())
    except (ValueError, AttributeError):
        low = esc_response.lower()
        if 'implement' in low:
            delg = 0
        elif 'escalat' in low:
            delg = 1
        else:
            print(f"Parse error (esc): {esc_response}", flush=True)
            delg = None

    trace = f"[PRED PROMPT]\n{pred_prompt}\n\n[PRED RESPONSE]\n{pred_response}\n\n[ESC PROMPT]\n{esc_prompt}\n\n[ESC RESPONSE]\n{esc_response}"
    return {"pred": pred, "del": delg, "full_prompt": pred_prompt, "response": pred_response, "trace": trace}

def get_llm_rf(base_prompt, rf_pred, rf_prob, model):
    rf_label = "keep the booking" if rf_pred == 1 else "cancel"
    rf_context = (
        f"{base_prompt}\n\n"
        f"A random forest trained on a similar dataset predicts Person 1 would {rf_label}, "
        f"based on a predicted probability of {rf_prob:.3f} that Person 1 would keep the booking."
    )

    # Step 1: Get prediction
    pred_prompt = rf_context + "\n\nRespond with a single digit ONLY. 1 (will keep booking) or 0 (will cancel) based on your prediction of Person 1's true decision."
    pred_response = llm(pred_prompt, model)
    try:
        pred = int(re.search(r'[01]', pred_response.strip()).group())
    except (ValueError, AttributeError):
        print(f"Parse error (pred): {pred_response}", flush=True)
        pred = None

    # Step 2: Get escalation decision
    esc_prompt = (
        f"{rf_context}\n\nYour prediction: {pred_response.strip()}\n\n"
        "Respond with a single digit ONLY.\n"
        "Output 0 to implement your prediction of Person 1's true decision, "
        "or 1 to escalate to Person 1 so they can implement their true decision.\n"
        "A decision is considered correct if it equals Person 1's true decision."
    )
    esc_response = llm(esc_prompt, model)
    try:
        delg = int(re.search(r'[01]', esc_response.strip()).group())
    except (ValueError, AttributeError):
        low = esc_response.lower()
        if 'implement' in low:
            delg = 0
        elif 'escalat' in low:
            delg = 1
        else:
            print(f"Parse error (esc): {esc_response}", flush=True)
            delg = None

    trace = f"[PRED PROMPT]\n{pred_prompt}\n\n[PRED RESPONSE]\n{pred_response}\n\n[ESC PROMPT]\n{esc_prompt}\n\n[ESC RESPONSE]\n{esc_response}"
    return {"pred": pred, "del": delg, "full_prompt": pred_prompt, "response": pred_response, "trace": trace}

def get_sequential_inference(base_prompt, model):
    try:
        thought = llm(
            f"{base_prompt}\n\n"
            "TASK: Predict Person 1's decision. Explain your reasoning in 1 sentence. "
            "Then conclude with exactly 'PREDICTION: 1' (keep booking) or 'PREDICTION: 0' (cancel).",
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
        'human_response': human_response,
        'prompt': base,
        'method': method,
        'model': model,
    }

    if method == "base":
        result = get_llm_base(base, model)
        trace = result.get("trace", f"[PROMPT]\n{result['full_prompt']}\n\n[RESPONSE]\n{result['response']}")
        return {**common, 'llm_prediction': result['pred'], 'llm_escalate': result['del'], 'trace': trace}
    elif method == "rf":
        rf_pred_val = row['rf_pred']
        rf_prob_val = row['rf_prob']
        result = get_llm_rf(base, rf_pred_val, rf_prob_val, model)
        trace = result.get("trace", f"[PROMPT]\n{result['full_prompt']}\n\n[RESPONSE]\n{result['response']}")
        return {**common, 'llm_prediction': result['pred'], 'llm_escalate': result['del'],
                'rf_pred': rf_pred_val, 'rf_prob': rf_prob_val, 'trace': trace}
    elif method == "adversarial":
        result = get_sequential_inference(base, model)
        trace = (f"[PROMPT]\n{base}\n\n[THOUGHT]\n{result['full_thought']}\n\n"
                 f"[CRITIQUE]\n{result['critique']}\n\n[DECISION PROMPT]\n{result['decision_prompt']}\n\n[DECISION]\n{result['decision']}")
        return {**common, 'llm_prediction': result['pred'], 'llm_escalate': result['del'],
                'llm_full_thought': result['full_thought'], 'llm_critique': result['critique'], 'trace': trace}

# --- Output ---
local_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../results/HotelBookings")
os.makedirs(local_dir, exist_ok=True)

def get_path(method, model):
    return os.path.join(local_dir, f'{method}_{model.split("/")[-1]}.csv')

df_existing = {}
for model, n in [(QWEN_MODEL, N_QWEN)]:
    if n > 0:
        for method in ["base", "rf", "adversarial"]:
            path = get_path(method, model)
            try:
                df_existing[(method, model)] = pd.read_csv(path)
            except FileNotFoundError:
                df_existing[(method, model)] = pd.DataFrame()

results = []
completed = 0
total = N_QWEN * (N_SAMPLES_BASE + N_SAMPLES_RF + N_SAMPLES_ADVERSARIAL)
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
        for method, n_samples in [("base", N_SAMPLES_BASE), ("rf", N_SAMPLES_RF), ("adversarial", N_SAMPLES_ADVERSARIAL)]:
            if n_samples > 0:
                sampled = random.sample(holdout_indices, n * n_samples)
                for idx in sampled:
                    jobs.append((idx, method, model))

print(f"Starting {total} jobs | Qwen {N_QWEN}x(b={N_SAMPLES_BASE}, r={N_SAMPLES_RF}, a={N_SAMPLES_ADVERSARIAL})", flush=True)
with ThreadPoolExecutor(max_workers=1) as executor:
    futures = [executor.submit(call_llm_tracked, idx, method, model) for idx, method, model in jobs]
    for f in as_completed(futures):
        f.result()

df_new = pd.DataFrame([r for r in results if r is not None])
for (method, model), group in df_new.groupby(['method', 'model']):
    path = get_path(method, model)
    print(f"Saved to {path}", flush=True)
    print(pd.read_csv(path)[['id', 'llm_prediction', 'human_response', 'llm_escalate', 'method', 'model']].to_string(), flush=True)
