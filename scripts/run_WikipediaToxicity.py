import os, re, datetime, threading, random, time
import numpy as np
import pandas as pd
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

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/WikipediaToxicity/Wikipedia Toxicity_data_data.csv")

# --- Load and aggregate data ---
print("Loading Wikipedia Toxicity data...", flush=True)
data = pd.read_csv(DATA_PATH)

data["comment"] = (
    data["comment"]
        .str.replace("NEWLINE_TOKEN", " \n ", regex=False)
        .str.replace("TAB_TOKEN", " \t ", regex=False)
)

data_agg = (
    data.groupby("rev_id", as_index=False)
        .agg(
            toxicity_score_mean=("toxicity_score", "mean"),
            comment=("comment", "first"),
            year=("year", "first"),
            ns=("ns", "first"),
            sample=("sample", "first"),
            split=("split", "first"),
        )
)
data_agg["toxicity"] = np.where(data_agg["toxicity_score_mean"] < 0, 1, 0)
data_agg = data_agg.reset_index(drop=True)

print(f"Loaded {len(data_agg)} comments. Toxicity rate: {data_agg['toxicity'].mean():.3f}", flush=True)

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
        f"You are helping a group of crowd-workers label Wikipedia discussion comments as toxic or not. "
        f"This comment needs to be checked: '{row['comment']}'."
    )

def get_llm_base(base_prompt, model):
    instructions = (
        "\n\nRespond with two numbers ONLY on two lines. "
        "First, respond 1 (toxic) or 0 (not toxic) based on your prediction of the crowd-workers' true aggregate decision.\n"
        "Second, respond 0 to implement your prediction of the crowd-workers' true aggregate decision, "
        "or 1 to delegate to the crowd-workers so they can implement their true aggregate decision.\n"
        "A decision is considered correct if it equals the crowd-workers' true aggregate decision."
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
    full_prompt = (
        base_prompt +
        "\n\nRespond with a single digit ONLY: 1 (toxic) or 0 (not toxic) for your prediction of the crowd-workers' true aggregate decision."
    )
    t0 = time.time()
    r = oai_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": full_prompt}],
        reasoning_effort="medium",
    )
    print(f"[{model.split(chr(47))[-1]} cot] {time.time()-t0:.1f}s", flush=True)
    response = r.choices[0].message.content.strip()
    reasoning_tokens = getattr(getattr(r.usage, 'completion_tokens_details', None), 'reasoning_tokens', None)
    pred_match = re.search(r'[01]', response)
    pred = int(pred_match.group()) if pred_match else None
    return {"full_prompt": full_prompt, "response": response, "pred": pred, "reasoning_tokens": reasoning_tokens}

def get_sequential_inference(base_prompt, model):
    try:
        thought = llm(
            f"{base_prompt}\n\n"
            "TASK: Predict the crowd-workers' aggregate decision. Explain your reasoning in 1 sentence. "
            "Then conclude with exactly 'PREDICTION: 1' (toxic) or 'PREDICTION: 0' (not toxic).",
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
            "Output 0 to implement your prediction, or 1 to delegate to the crowd-workers.\n"
            "A decision is correct if it equals the crowd-workers' true aggregate decision."
        )
        decision = llm(decision_prompt, model)
        del_match = re.search(r'[01]', decision.strip())
        final_del = int(del_match.group()) if del_match else 1
        return {"full_thought": thought, "pred": pred, "critique": critique, "decision_prompt": decision_prompt, "decision": decision, "del": final_del}
    except Exception as e:
        return {"full_thought": str(e), "pred": None, "critique": None, "decision": None, "del": None}

def call_llm(row_idx, method, model):
    row = data_agg.loc[row_idx]
    base = create_prompt_base(row)
    human_response = int(row['toxicity'])

    common = {
        'rev_id': row['rev_id'],
        'human_response': human_response,
        'prompt': base,
        'method': method,
        'model': model,
    }

    if method == "base":
        result = get_llm_base(base, model)
        trace = f"[PROMPT]\n{result['full_prompt']}\n\n[RESPONSE]\n{result['response']}"
        return {**common, 'llm_prediction': result['pred'], 'llm_delegate': result['del'], 'trace': trace}
    elif method == "cot":
        result = get_llm_cot(base, model)
        trace = f"[PROMPT]\n{result['full_prompt']}\n\n[RESPONSE]\n{result['response']}"
        return {**common, 'llm_prediction': result['pred'], 'llm_delegate': None, 'reasoning_tokens': result['reasoning_tokens'], 'trace': trace}
    elif method == "auditor":
        result = get_sequential_inference(base, model)
        trace = (f"[PROMPT]\n{base}\n\n[THOUGHT]\n{result['full_thought']}\n\n"
                 f"[CRITIQUE]\n{result['critique']}\n\n[DECISION PROMPT]\n{result['decision_prompt']}\n\n[DECISION]\n{result['decision']}")
        return {**common, 'llm_prediction': result['pred'], 'llm_delegate': result['del'],
                'llm_full_thought': result['full_thought'], 'llm_critique': result['critique'], 'trace': trace}

# --- Output ---
local_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../results/WikipediaToxicity")
os.makedirs(local_dir, exist_ok=True)

def get_path(method, model):
    return os.path.join(local_dir, f'{method}_{model.split("/")[-1]}.csv')

df_existing = {}
for model, n in [(OAI_MODEL, N_OAI), (OAI_MODEL_NANO, N_NANO), (QWEN_MODEL, N_QWEN), (QWEN_MODEL_MED, N_QWEN_MED), (QWEN_MODEL_LARGE, N_QWEN_LARGE), (QWEN_MODEL_XL, N_QWEN_XL), (GLM_MODEL, N_GLM), (DEEPSEEK_MODEL, N_DEEPSEEK)]:
    if n > 0:
        for method in ["base", "auditor", "cot"]:
            path = get_path(method, model)
            try:
                df_existing[(method, model)] = pd.read_csv(path)
            except FileNotFoundError:
                df_existing[(method, model)] = pd.DataFrame()

results = []
completed = 0
total = (N_OAI + N_NANO + N_QWEN + N_QWEN_MED + N_QWEN_LARGE + N_QWEN_XL + N_GLM + N_DEEPSEEK) * (N_SAMPLES_BASE + N_SAMPLES_AUDITOR) + N_NANO * N_SAMPLES_COT
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
all_indices = list(data_agg.index)
jobs = []
for model, n in [(OAI_MODEL, N_OAI), (OAI_MODEL_NANO, N_NANO), (QWEN_MODEL, N_QWEN), (QWEN_MODEL_MED, N_QWEN_MED), (QWEN_MODEL_LARGE, N_QWEN_LARGE), (QWEN_MODEL_XL, N_QWEN_XL), (GLM_MODEL, N_GLM), (DEEPSEEK_MODEL, N_DEEPSEEK)]:
    if n > 0:
        methods = [("base", N_SAMPLES_BASE), ("auditor", N_SAMPLES_AUDITOR)]
        if model == OAI_MODEL_NANO:
            methods.append(("cot", N_SAMPLES_COT))
        for method, n_samples in methods:
            if n_samples > 0:
                sampled = random.sample(all_indices, n * n_samples)
                for idx in sampled:
                    jobs.append((idx, method, model))

print(f"Starting {total} jobs | OAI {N_OAI}x(b={N_SAMPLES_BASE}, a={N_SAMPLES_AUDITOR}) | Nano {N_NANO}x(b={N_SAMPLES_BASE}, c={N_SAMPLES_COT}, a={N_SAMPLES_AUDITOR}) | Qwen {N_QWEN}x | QwenMed {N_QWEN_MED}x | QwenLarge {N_QWEN_LARGE}x | QwenXL {N_QWEN_XL}x | GLM {N_GLM}x | DeepSeek {N_DEEPSEEK}x", flush=True)
with ThreadPoolExecutor(max_workers=5) as executor:
    futures = [executor.submit(call_llm_tracked, idx, method, model) for idx, method, model in jobs]
    for f in as_completed(futures):
        f.result()

df_new = pd.DataFrame([r for r in results if r is not None])
for (method, model), group in df_new.groupby(['method', 'model']):
    path = get_path(method, model)
    print(f"Saved to {path}", flush=True)
    print(pd.read_csv(path)[['rev_id', 'llm_prediction', 'human_response', 'llm_delegate', 'method', 'model']].to_string(), flush=True)
