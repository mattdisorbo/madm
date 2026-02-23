import os, re, datetime, threading, random
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import openai

OAI_MODEL        = "gpt-5-mini-2025-08-07"
OAI_MODEL_NANO   = "gpt-5-nano-2025-08-07"
QWEN_MODEL       = "Qwen/Qwen2.5-1.5B-Instruct"
QWEN_MODEL_MED   = "Qwen/Qwen2.5-3B-Instruct"
QWEN_MODEL_LARGE = "Qwen/Qwen2.5-7B-Instruct"
GLM_MODEL        = "THUDM/glm-4-9b-chat-hf"

N_SAMPLES_BASE    = 10
N_SAMPLES_AUDITOR = 10
N_OAI        = 0
N_NANO       = 0
N_QWEN       = 0
N_QWEN_MED   = 0
N_QWEN_LARGE = 1
N_GLM        = 1

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/FEVEROUS/feverous_train_challenges.jsonl")

# --- Load and clean data ---
print("Loading FEVEROUS data...", flush=True)
data = pd.read_json(DATA_PATH, lines=True)
data = data.replace('', pd.NA).dropna(how='all')
data = data[data["label"] != "NOT ENOUGH INFO"]
data["supports"] = data["label"].map({"SUPPORTS": 1, "REFUTES": 0})
data = data.reset_index(drop=True)

print(f"Loaded {len(data)} claims. Support rate: {data['supports'].mean():.3f}", flush=True)

# --- Local model setup ---
local_pipes = {}
local_locks = {}

for _m, _n in [(QWEN_MODEL, N_QWEN), (QWEN_MODEL_MED, N_QWEN_MED), (QWEN_MODEL_LARGE, N_QWEN_LARGE), (GLM_MODEL, N_GLM)]:
    if _n > 0:
        from transformers import pipeline
        print(f"Loading {_m}...", flush=True)
        local_pipes[_m] = pipeline("text-generation", model=_m, torch_dtype="auto", device_map="auto", trust_remote_code=True)
        local_locks[_m] = threading.Lock()
        print(f"{_m} loaded.", flush=True)

def llm(prompt, model):
    if model in local_pipes:
        with local_locks[model]:
            out = local_pipes[model]([{"role": "user", "content": prompt}], max_new_tokens=2048)
        return out[0]["generated_text"][-1]["content"]
    else:
        r = openai.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}])
        return r.choices[0].message.content.strip()

def create_prompt_base(row):
    return (
        f"You are helping Person 1 with fact-checking. "
        f"This statement needs to be checked: {row['claim']}."
    )

def get_llm_base(base_prompt, model):
    instructions = (
        "\n\nRespond with two numbers ONLY on two lines. "
        "First, respond 1 (support) or 0 (refute) based on your prediction of Person 1's true decision.\n"
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
            "Then conclude with exactly 'PREDICTION: 1' (support) or 'PREDICTION: 0' (refute).",
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
    human_response = int(row['supports'])

    common = {
        'id': row['id'],
        'claim': row['claim'],
        'human_response': human_response,
        'prompt': base,
        'method': method,
        'model': model,
    }

    if method == "base":
        result = get_llm_base(base, model)
        trace = f"[PROMPT]\n{result['full_prompt']}\n\n[RESPONSE]\n{result['response']}"
        return {**common, 'llm_prediction': result['pred'], 'llm_delegate': result['del'], 'trace': trace}
    elif method == "auditor":
        result = get_sequential_inference(base, model)
        trace = (f"[PROMPT]\n{base}\n\n[THOUGHT]\n{result['full_thought']}\n\n"
                 f"[CRITIQUE]\n{result['critique']}\n\n[DECISION PROMPT]\n{result['decision_prompt']}\n\n[DECISION]\n{result['decision']}")
        return {**common, 'llm_prediction': result['pred'], 'llm_delegate': result['del'],
                'llm_full_thought': result['full_thought'], 'llm_critique': result['critique'], 'trace': trace}

# --- Output ---
local_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../results/FEVEROUS")
os.makedirs(local_dir, exist_ok=True)

def get_path(method, model):
    return os.path.join(local_dir, f'{method}_{model.split("/")[-1]}.csv')

df_existing = {}
for model, n in [(OAI_MODEL, N_OAI), (OAI_MODEL_NANO, N_NANO), (QWEN_MODEL, N_QWEN), (QWEN_MODEL_MED, N_QWEN_MED), (QWEN_MODEL_LARGE, N_QWEN_LARGE), (GLM_MODEL, N_GLM)]:
    if n > 0:
        for method in ["base", "auditor"]:
            path = get_path(method, model)
            try:
                df_existing[(method, model)] = pd.read_csv(path)
            except FileNotFoundError:
                df_existing[(method, model)] = pd.DataFrame()

results = []
completed = 0
total = (N_OAI + N_NANO + N_QWEN + N_QWEN_MED + N_QWEN_LARGE + N_GLM) * (N_SAMPLES_BASE + N_SAMPLES_AUDITOR)
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
all_indices = list(data.index)
jobs = []
for model, n in [(OAI_MODEL, N_OAI), (OAI_MODEL_NANO, N_NANO), (QWEN_MODEL, N_QWEN), (QWEN_MODEL_MED, N_QWEN_MED), (QWEN_MODEL_LARGE, N_QWEN_LARGE), (GLM_MODEL, N_GLM)]:
    if n > 0:
        for method, n_samples in [("base", N_SAMPLES_BASE), ("auditor", N_SAMPLES_AUDITOR)]:
            if n_samples > 0:
                sampled = random.sample(all_indices, n * n_samples)
                for idx in sampled:
                    jobs.append((idx, method, model))

print(f"Starting {total} jobs | OAI {N_OAI}x(b={N_SAMPLES_BASE}, a={N_SAMPLES_AUDITOR}) | Nano {N_NANO}x(b={N_SAMPLES_BASE}, a={N_SAMPLES_AUDITOR}) | Qwen {N_QWEN}x | QwenMed {N_QWEN_MED}x | QwenLarge {N_QWEN_LARGE}x | GLM {N_GLM}x(b={N_SAMPLES_BASE}, a={N_SAMPLES_AUDITOR})", flush=True)
with ThreadPoolExecutor(max_workers=5) as executor:
    futures = [executor.submit(call_llm_tracked, idx, method, model) for idx, method, model in jobs]
    for f in as_completed(futures):
        f.result()

df_new = pd.DataFrame([r for r in results if r is not None])
for (method, model), group in df_new.groupby(['method', 'model']):
    path = get_path(method, model)
    print(f"Saved to {path}", flush=True)
    print(pd.read_csv(path)[['id', 'llm_prediction', 'human_response', 'llm_delegate', 'method', 'model']].to_string(), flush=True)
