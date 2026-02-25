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

N_SAMPLES_BASE    = 0
N_SAMPLES_RF      = 0
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
    instructions = (
        "\n\nRespond with two numbers ONLY on two lines. "
        "First, respond 1 (swerve) or 0 (do not swerve) based on your prediction of Person 1's true decision.\n"
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
    rf_label = "swerve" if rf_pred == 1 else "not swerve"
    instructions = (
        f"A random forest trained on a similar dataset predicts Person 1 would {rf_label}, "
        f"based on a predicted probability of {rf_prob:.3f} that Person 1 would swerve. "
        "\n\nRespond with two numbers ONLY on two lines. "
        "First, respond 1 (swerve) or 0 (do not swerve) based on your prediction of Person 1's true decision.\n"
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
        "First, respond 1 (swerve) or 0 (do not swerve) based on your prediction of Person 1's true decision.\n"
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
        trace = f"[PROMPT]\n{result['full_prompt']}\n\n[RESPONSE]\n{result['response']}"
        return {**common, 'llm_prediction': result['pred'], 'llm_delegate': result['del'], 'trace': trace}
    elif method == "rf":
        rf_pred_val = row['rf_pred']
        rf_prob_val = row['rf_prob']
        result = get_llm_rf(base, rf_pred_val, rf_prob_val, model)
        trace = f"[PROMPT]\n{result['full_prompt']}\n\n[RESPONSE]\n{result['response']}"
        return {**common, 'llm_prediction': result['pred'], 'llm_delegate': result['del'],
                'rf_pred': rf_pred_val, 'rf_prob': rf_prob_val, 'trace': trace}
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
local_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../results/MoralMachine")
os.makedirs(local_dir, exist_ok=True)

def get_path(method, model):
    return os.path.join(local_dir, f'{method}_{model.split("/")[-1]}.csv')

df_existing = {}
for model, n in [(OAI_MODEL, N_OAI), (OAI_MODEL_NANO, N_NANO), (QWEN_MODEL, N_QWEN), (QWEN_MODEL_MED, N_QWEN_MED), (QWEN_MODEL_LARGE, N_QWEN_LARGE), (QWEN_MODEL_XL, N_QWEN_XL), (GLM_MODEL, N_GLM), (DEEPSEEK_MODEL, N_DEEPSEEK)]:
    if n > 0:
        for method in ["base", "rf", "auditor", "cot"]:
            path = get_path(method, model)
            try:
                df_existing[(method, model)] = pd.read_csv(path)
            except FileNotFoundError:
                df_existing[(method, model)] = pd.DataFrame()

results = []
completed = 0
total = (N_OAI + N_NANO + N_QWEN + N_QWEN_MED + N_QWEN_LARGE + N_QWEN_XL + N_GLM + N_DEEPSEEK) * (N_SAMPLES_BASE + N_SAMPLES_RF + N_SAMPLES_AUDITOR) + N_NANO * N_SAMPLES_COT
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
        methods = [("base", N_SAMPLES_BASE), ("rf", N_SAMPLES_RF), ("auditor", N_SAMPLES_AUDITOR)]
        if model == OAI_MODEL_NANO:
            methods.append(("cot", N_SAMPLES_COT))
        for method, n_samples in methods:
            if n_samples > 0:
                sampled = random.sample(holdout_indices, n * n_samples)
                for idx in sampled:
                    jobs.append((idx, method, model))

print(f"Starting {total} jobs | OAI {N_OAI}x(b={N_SAMPLES_BASE}, r={N_SAMPLES_RF}, a={N_SAMPLES_AUDITOR}) | Nano {N_NANO}x(b={N_SAMPLES_BASE}, c={N_SAMPLES_COT}, r={N_SAMPLES_RF}, a={N_SAMPLES_AUDITOR}) | Qwen {N_QWEN}x | QwenMed {N_QWEN_MED}x | QwenLarge {N_QWEN_LARGE}x | QwenXL {N_QWEN_XL}x | GLM {N_GLM}x | DeepSeek {N_DEEPSEEK}x", flush=True)
with ThreadPoolExecutor(max_workers=5) as executor:
    futures = [executor.submit(call_llm_tracked, idx, method, model) for idx, method, model in jobs]
    for f in as_completed(futures):
        f.result()

df_new = pd.DataFrame([r for r in results if r is not None])
for (method, model), group in df_new.groupby(['method', 'model']):
    path = get_path(method, model)
    print(f"Saved to {path}", flush=True)
    print(pd.read_csv(path)[['ResponseID', 'llm_prediction', 'human_response', 'llm_delegate', 'method', 'model']].to_string(), flush=True)
