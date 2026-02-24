import os, re, datetime, threading
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
QWEN_MODEL       = "Qwen/Qwen2.5-1.5B-Instruct"
QWEN_MODEL_MED   = "Qwen/Qwen2.5-3B-Instruct"
QWEN_MODEL_LARGE = "Qwen/Qwen2.5-7B-Instruct"
GLM_MODEL        = "THUDM/glm-4-9b-chat-hf"
DEEPSEEK_MODEL   = "deepseek-chat"

N_SAMPLES_BASE    = 10
N_SAMPLES_AUDITOR = 10
N_OAI        = 0
N_NANO       = 0
N_QWEN       = 0
N_QWEN_MED   = 0
N_QWEN_LARGE = 1
N_GLM        = 1
N_DEEPSEEK   = 0

df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/AIME_Dataset_1983_2024.csv"))

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
    elif model == DEEPSEEK_MODEL:
        r = deepseek_client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}])
        return r.choices[0].message.content.strip()
    else:
        r = oai_client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}])
        return r.choices[0].message.content.strip()

def create_prompt_base(row):
    return f"You are helping Person 1 solve the following math problem: {row['Question']}."

def get_llm_base(prompt, model):
    instructions = (
        "\n\nRespond with two numbers ONLY on two lines. "
        "First, your integer solution. "
        "Second, 0 to implement your solution or 1 to delegate to Person 1."
    )
    full_prompt = prompt + instructions
    response = llm(full_prompt, model)
    lines = [l.strip() for l in response.strip().split('\n') if l.strip()]
    try:
        pred = int(re.search(r'\d+', lines[0]).group()) if len(lines) > 0 else None
        delg = int(re.search(r'[01]', lines[-1]).group()) if len(lines) > 1 else None
        return {"pred": pred, "del": delg, "full_prompt": full_prompt, "response": response}
    except (ValueError, IndexError, AttributeError):
        print(f"Parse error: {response}", flush=True)
        return {"pred": None, "del": None, "full_prompt": full_prompt, "response": response}

def get_sequential_inference(scenario, model):
    try:
        thought = llm(f"{scenario}\n\nSolve the math problem step by step, "
                      "then conclude with 'SOLUTION: <integer>'.", model)
        preds = re.findall(r'SOLUTION:\s*(\d+)', thought)
        if not preds:
            preds = re.findall(r'\\boxed\{(\d+)\}', thought)
        pred = preds[-1] if preds else "0"
        critique = llm(f"PROBLEM: {scenario}\n\nPROPOSED SOLUTION: {thought}\n\nCritique this reasoning in 1 sentence.", model)
        decision_prompt = (f"PROBLEM: {scenario}\nSOLUTION AND REASONING: {thought}\nCRITIQUE: {critique}\n\n"
                           "Implement your solution (0) or delegate to Person 1 (1). Respond with the digit ONLY.")
        decision = llm(decision_prompt, model)
        del_match = re.search(r'[01]', decision)
        delg = int(del_match.group()) if del_match else None
        return {"full_thought": thought, "pred": pred, "critique": critique, "decision_prompt": decision_prompt, "decision": decision, "del": delg}
    except Exception as e:
        return {"full_thought": str(e), "pred": "Err", "critique": "Err", "decision": "Err", "del": "1"}

def call_llm(idx, row, method, model):
    base = create_prompt_base(row)
    if method == "base":
        result = get_llm_base(base, model)
        trace = f"[PROMPT]\n{result['full_prompt']}\n\n[RESPONSE]\n{result['response']}"
        return {**row, "prompt": base, "llm_prediction": result["pred"],
                "llm_delegate": result["del"], "solution": row["Answer"],
                "method": method, "model": model, "trace": trace}
    elif method == "auditor":
        result = get_sequential_inference(base, model)
        trace = (f"[PROMPT]\n{base}\n\n"
                 f"[THOUGHT]\n{result['full_thought']}\n\n"
                 f"[CRITIQUE]\n{result['critique']}\n\n"
                 f"[DECISION PROMPT]\n{result['decision_prompt']}\n\n"
                 f"[DECISION]\n{result['decision']}")
        return {**row, "prompt": base, "llm_full_thought": result["full_thought"],
                "llm_prediction": result["pred"], "llm_critique": result["critique"],
                "llm_delegate": result["del"], "solution": row["Answer"],
                "method": method, "model": model, "trace": trace}

# --- Output ---
local_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../results/AIME")
os.makedirs(local_dir, exist_ok=True)

def get_path(method, model):
    return os.path.join(local_dir, f'{method}_{model.split("/")[-1]}.csv')

# Load existing data per (method, model) combination
df_existing = {}
for model, n in [(OAI_MODEL, N_OAI), (OAI_MODEL_NANO, N_NANO), (QWEN_MODEL, N_QWEN), (QWEN_MODEL_MED, N_QWEN_MED), (QWEN_MODEL_LARGE, N_QWEN_LARGE), (GLM_MODEL, N_GLM), (DEEPSEEK_MODEL, N_DEEPSEEK)]:
    if n > 0:
        for method in ["base", "auditor"]:
            path = get_path(method, model)
            try:
                df_existing[(method, model)] = pd.read_csv(path)
            except FileNotFoundError:
                df_existing[(method, model)] = pd.DataFrame()

results = []
completed = 0
total = (N_OAI + N_NANO + N_QWEN + N_QWEN_MED + N_QWEN_LARGE + N_GLM + N_DEEPSEEK) * (N_SAMPLES_BASE + N_SAMPLES_AUDITOR)
save_lock = threading.Lock()

def save_progress():
    df_new = pd.DataFrame(results)
    df_new['timestamp'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for (method, model), group in df_new.groupby(['method', 'model']):
        path = get_path(method, model)
        pd.concat([df_existing.get((method, model), pd.DataFrame()), group], ignore_index=True).to_csv(path, index=False)

def call_llm_tracked(idx, row, method, model):
    global completed
    result = call_llm(idx, row, method, model)
    with save_lock:
        completed += 1
        results.append(result)
        print(f"[{completed}/{total}] Done: row {idx} ({method}, {model})", flush=True)
        save_progress()
    return result

# --- Build jobs ---
jobs = []
for model, n in [(OAI_MODEL, N_OAI), (OAI_MODEL_NANO, N_NANO), (QWEN_MODEL, N_QWEN), (QWEN_MODEL_MED, N_QWEN_MED), (QWEN_MODEL_LARGE, N_QWEN_LARGE), (GLM_MODEL, N_GLM), (DEEPSEEK_MODEL, N_DEEPSEEK)]:
    if n > 0:
        for method, n_samples in [("base", N_SAMPLES_BASE), ("auditor", N_SAMPLES_AUDITOR)]:
            if n_samples > 0:
                for idx, row in df.sample(n=n * n_samples).iterrows():
                    jobs.append((idx, row, method, model))

print(f"Starting {total} jobs | OAI {N_OAI}x(b={N_SAMPLES_BASE}, a={N_SAMPLES_AUDITOR}) | Nano {N_NANO}x(b={N_SAMPLES_BASE}, a={N_SAMPLES_AUDITOR}) | Qwen {N_QWEN}x | QwenMed {N_QWEN_MED}x | QwenLarge {N_QWEN_LARGE}x | GLM {N_GLM}x(b={N_SAMPLES_BASE}, a={N_SAMPLES_AUDITOR}) | DeepSeek {N_DEEPSEEK}x(b={N_SAMPLES_BASE}, a={N_SAMPLES_AUDITOR})", flush=True)
with ThreadPoolExecutor(max_workers=5) as executor:
    futures = [executor.submit(call_llm_tracked, idx, row, method, model) for idx, row, method, model in jobs]
    for f in as_completed(futures):
        f.result()

df_new = pd.DataFrame(results)
for (method, model), group in df_new.groupby(['method', 'model']):
    path = get_path(method, model)
    print(f"Saved to {path}", flush=True)
    print(pd.read_csv(path)[['ID', 'llm_prediction', 'solution', 'llm_delegate', 'method', 'model']].to_string(), flush=True)
