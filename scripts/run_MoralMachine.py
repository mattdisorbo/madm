import os, re, sys, datetime, threading, random
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor, as_completed
import openai

OAI_MODEL      = "gpt-5-mini-2025-08-07"
OAI_MODEL_NANO = "gpt-5-nano-2025-08-07"
QWEN_MODEL     = "Qwen/Qwen3.5-35B-A3B"
if len(sys.argv) > 1:
    QWEN_MODEL = sys.argv[1]

N_SAMPLES_BASE    = 50
N_SAMPLES_RF = 50
N_SAMPLES_ADVERSARIAL = 50
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

# --- Train Random Forest ---
print("Training Random Forest...", flush=True)
features = [
    'Intervention_0', 'Intervention_1', 'PedPed_0', 'PedPed_1',
    'Barrier_0', 'Barrier_1', 'CrossingSignal_0', 'CrossingSignal_1',
    'AttributeLevel_0', 'AttributeLevel_1', 'NumberOfCharacters_0',
    'NumberOfCharacters_1', 'UserCountry3_0', 'Review_age_0',
    'Review_education_0', 'Review_gender_0', 'Review_income_0',
    'Review_political_0', 'Review_religious_0'
]
target = 'Saved_0'

df_clean = df_wide[features + [target]].copy()
df_clean['Review_age_0'] = pd.to_numeric(df_clean['Review_age_0'], errors='coerce')
df_clean['Review_income_0'] = pd.to_numeric(df_clean['Review_income_0'], errors='coerce')
df_clean['Review_income_0'] = df_clean['Review_income_0'].clip(lower=0, upper=1e6)
df_clean = df_clean[(df_clean['Review_age_0'] >= 0) & (df_clean['Review_age_0'] <= 120)]
df_clean.replace([np.inf, -np.inf], np.nan, inplace=True)
df_clean.dropna(inplace=True)

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

df_wide['rf_pred'] = np.nan
df_wide['rf_prob'] = np.nan
df_wide.loc[holdout_idx, 'rf_pred'] = rf_preds
df_wide.loc[holdout_idx, 'rf_prob'] = rf_probs

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
    "under5000": "Under $5,000", "5000": "$5,000–$10,000", "10000": "$10,001–$15,000",
    "15000": "$15,001–$25,000", "25000": "$25,001–$35,000", "35000": "$35,001–$50,000",
    "50000": "$50,001–$80,000", "80000": "$80,001–$100,000", "above100000": "Over $100,000"
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

def get_llm_base(base_prompt, model):
    # Step 1: Get prediction with reasoning
    pred_prompt = (
        f"{base_prompt}\n\n"
        "TASK: Predict Person 1's decision. Explain your reasoning in 1 sentence. "
        "Then conclude with exactly 'PREDICTION: 1' (swerve) or 'PREDICTION: 0' (do not swerve)."
    )
    pred_response = llm(pred_prompt, model)
    preds = re.findall(r'PREDICTION:\s*([01])', pred_response)
    pred = int(preds[-1]) if preds else None
    if pred is None:
        print(f"Parse error (pred): {pred_response}", flush=True)

    # Step 2: Get escalation decision (with reasoning context, no critique)
    esc_prompt = (
        f"SCENARIO:\n{base_prompt}\n\nPREDICTION & REASONING:\n{pred_response}\n\n"
        "TASK: Respond with a digit ONLY.\n"
        "Output 0 to implement your prediction, or 1 to escalate to Person 1.\n"
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
    rf_label = "swerve" if rf_pred == 1 else "not swerve"
    rf_context = (
        f"{base_prompt}\n\n"
        f"A random forest trained on a similar dataset predicts Person 1 would {rf_label}, "
        f"based on a predicted probability of {rf_prob:.3f} that Person 1 would swerve."
    )

    # Step 1: Get prediction
    pred_prompt = rf_context + "\n\nRespond with a single digit ONLY. 1 (swerve) or 0 (do not swerve) based on your prediction of Person 1's true decision."
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
            "Then conclude with exactly 'PREDICTION: 1' (swerve) or 'PREDICTION: 0' (do not swerve).",
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
            "A decision is considered correct if it equals Person 1's true decision."
        )
        decision = llm(decision_prompt, model)
        del_match = re.search(r'[01]', decision.strip())
        final_del = int(del_match.group()) if del_match else 1
        return {"full_thought": thought, "pred": pred, "critique": critique, "decision_prompt": decision_prompt, "decision": decision, "del": final_del}
    except Exception as e:
        return {"full_thought": str(e), "pred": None, "critique": None, "decision": None, "del": None}

def call_llm(row_idx, method, model):
    row = df_wide.loc[row_idx]
    base = create_prompt_base(row)
    human_response = int(row['Saved_0'])

    common = {
        'ResponseID': row['ResponseID'],
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
local_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../results/MoralMachine")
os.makedirs(local_dir, exist_ok=True)

def get_path(method, model):
    return os.path.join(local_dir, f'{method}_{model.split("/")[-1]}.csv')

df_existing = {}
for model, n in [(OAI_MODEL, N_OAI), (OAI_MODEL_NANO, N_NANO), (QWEN_MODEL, N_QWEN)]:
    if n > 0:
        for method in ["base", "rf", "adversarial"]:
            path = get_path(method, model)
            try:
                df_existing[(method, model)] = pd.read_csv(path)
            except FileNotFoundError:
                df_existing[(method, model)] = pd.DataFrame()

results = []
completed = 0
total = (N_OAI + N_NANO + N_QWEN) * (N_SAMPLES_BASE + N_SAMPLES_RF + N_SAMPLES_ADVERSARIAL)
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
        for method, n_samples in [("base", N_SAMPLES_BASE), ("rf", N_SAMPLES_RF), ("adversarial", N_SAMPLES_ADVERSARIAL)]:
            if n_samples > 0:
                sampled = random.sample(holdout_indices, n * n_samples)
                for idx in sampled:
                    jobs.append((idx, method, model))

print(f"Starting {total} jobs | OAI {N_OAI}x(b={N_SAMPLES_BASE}, r={N_SAMPLES_RF}, a={N_SAMPLES_ADVERSARIAL}) | Nano {N_NANO}x(b={N_SAMPLES_BASE}, r={N_SAMPLES_RF}, a={N_SAMPLES_ADVERSARIAL}) | Qwen {N_QWEN}x(b={N_SAMPLES_BASE}, r={N_SAMPLES_RF}, a={N_SAMPLES_ADVERSARIAL})", flush=True)
with ThreadPoolExecutor(max_workers=5) as executor:
    futures = [executor.submit(call_llm_tracked, idx, method, model) for idx, method, model in jobs]
    for f in as_completed(futures):
        f.result()

df_new = pd.DataFrame([r for r in results if r is not None])
for (method, model), group in df_new.groupby(['method', 'model']):
    path = get_path(method, model)
    print(f"Saved to {path}", flush=True)
    print(pd.read_csv(path)[['ResponseID', 'llm_prediction', 'human_response', 'llm_escalate', 'method', 'model']].to_string(), flush=True)
