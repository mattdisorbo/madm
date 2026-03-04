import os, re, sys, datetime, threading
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import openai

OAI_MODEL      = "gpt-5-mini-2025-08-07"
OAI_MODEL_NANO = "gpt-5-nano-2025-08-07"
QWEN_MODEL     = "Qwen/Qwen3.5-35B-A3B"
if len(sys.argv) > 1:
    QWEN_MODEL = sys.argv[1]

N_SAMPLES_BASE    = int(os.environ.get("N_SAMPLES_BASE", 50))
N_SAMPLES_ADVERSARIAL = int(os.environ.get("N_SAMPLES_ADVERSARIAL", 50))
N_OAI  = int(os.environ.get("N_OAI", 0))
N_NANO = int(os.environ.get("N_NANO", 0))
N_QWEN = int(os.environ.get("N_QWEN", 1))

df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/AIME_Dataset_1983_2024.csv"))

# --- Qwen setup ---
qwen_pipe = None
qwen_lock = threading.Lock()

if N_QWEN > 0:
    from transformers import pipeline
    print(f"Loading {QWEN_MODEL}...")
    qwen_pipe = pipeline("text-generation", model=QWEN_MODEL, torch_dtype="bfloat16", device_map="auto")
    print("Qwen loaded.")

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

def create_prompt_base(row):
    return f"You are helping Person 1 solve the following math problem: {row['Question']}."

def get_llm_base(prompt, model):
    # Step 1: Get prediction only (no reasoning)
    pred_prompt = (
        f"{prompt}\n\n"
        "TASK: Respond with a single integer ONLY. "
        "What is the numerical answer to this math problem?"
    )
    pred_response = llm(pred_prompt, model)
    preds = re.findall(r'(\d+)', pred_response.strip())
    pred = int(preds[0]) if preds else None
    if pred is None:
        print(f"Parse error (pred): {pred_response}")

    # Step 2: Get escalation decision
    esc_prompt = (
        f"PROBLEM:\n{prompt}\n\nYour answer: {pred_response.strip()}\n\n"
        "TASK: Respond with a digit ONLY.\n"
        "Output 0 to implement your answer, or 1 to escalate to Person 1.\n"
        "A decision is correct if it equals Person 1's true answer."
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
            print(f"Parse error (esc): {esc_response}")
            delg = None

    trace = f"[PRED PROMPT]\n{pred_prompt}\n\n[PRED RESPONSE]\n{pred_response}\n\n[ESC PROMPT]\n{esc_prompt}\n\n[ESC RESPONSE]\n{esc_response}"
    return {"pred": pred, "del": delg, "full_prompt": pred_prompt, "response": pred_response, "trace": trace}

def get_sequential_inference(scenario, model):
    try:
        thought = llm(f"{scenario}\n\nSolve the math problem step by step, "
                      "then conclude with 'SOLUTION: <integer>'.", model)
        preds = re.findall(r'SOLUTION:\s*(\d+)', thought)
        if not preds:
            preds = re.findall(r'\\boxed\{(\d+)\}', thought)
        if not preds:
            preds = re.findall(r'\$\s*(\d+)\s*\$\s*$', thought, re.MULTILINE)
        if not preds:
            preds = re.findall(r'(?:answer|result|equal(?:s)?|is)\s*(?:is\s*)?[\$\s]*(\d+)', thought, re.IGNORECASE)
        pred = preds[-1] if preds else "0"
        critique = llm(f"PROBLEM: {scenario}\n\nPROPOSED SOLUTION: {thought}\n\n"
                       "TASK: Give one reason why this solution should be escalated to Person 1 instead of being implemented automatically.", model)
        decision_prompt = (f"PROBLEM: {scenario}\nSOLUTION AND REASONING: {thought}\nCRITIQUE: {critique}\n\n"
                           "TASK: Respond with a digit ONLY.\n"
                           "Output 0 to implement your solution, or 1 to escalate to Person 1.\n"
                           "A decision is correct if it equals Person 1's true answer.")
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
        trace = result.get("trace", f"[PROMPT]\n{result['full_prompt']}\n\n[RESPONSE]\n{result['response']}")
        return {**row, "prompt": base, "llm_prediction": result["pred"],
                "llm_escalate": result["del"], "solution": row["Answer"],
                "method": method, "model": model, "trace": trace}
    elif method == "adversarial":
        result = get_sequential_inference(base, model)
        trace = (f"[PROMPT]\n{base}\n\n"
                 f"[THOUGHT]\n{result['full_thought']}\n\n"
                 f"[CRITIQUE]\n{result['critique']}\n\n"
                 f"[DECISION PROMPT]\n{result['decision_prompt']}\n\n"
                 f"[DECISION]\n{result['decision']}")
        return {**row, "prompt": base, "llm_full_thought": result["full_thought"],
                "llm_prediction": result["pred"], "llm_critique": result["critique"],
                "llm_escalate": result["del"], "solution": row["Answer"],
                "method": method, "model": model, "trace": trace}

# --- Output ---
local_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../results/AIME")
os.makedirs(local_dir, exist_ok=True)

def get_path(method, model):
    return os.path.join(local_dir, f'{method}_{model.split("/")[-1]}.csv')

# Load existing data per (method, model) combination
df_existing = {}
for model, n in [(OAI_MODEL, N_OAI), (OAI_MODEL_NANO, N_NANO), (QWEN_MODEL, N_QWEN)]:
    if n > 0:
        for method in ["base", "adversarial"]:
            path = get_path(method, model)
            try:
                df_existing[(method, model)] = pd.read_csv(path)
            except FileNotFoundError:
                df_existing[(method, model)] = pd.DataFrame()

results = []
completed = 0
total = (N_OAI + N_NANO + N_QWEN) * (N_SAMPLES_BASE + N_SAMPLES_ADVERSARIAL)
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
        print(f"[{completed}/{total}] Done: row {idx} ({method}, {model})")
        save_progress()
    return result

# --- Build jobs ---
jobs = []
for model, n in [(OAI_MODEL, N_OAI), (OAI_MODEL_NANO, N_NANO), (QWEN_MODEL, N_QWEN)]:
    if n > 0:
        for method, n_samples in [("base", N_SAMPLES_BASE), ("adversarial", N_SAMPLES_ADVERSARIAL)]:
            if n_samples > 0:
                for idx, row in df.sample(n=n * n_samples).iterrows():
                    jobs.append((idx, row, method, model))

print(f"Starting {total} jobs | OAI {N_OAI}x(b={N_SAMPLES_BASE}, a={N_SAMPLES_ADVERSARIAL}) | Nano {N_NANO}x(b={N_SAMPLES_BASE}, a={N_SAMPLES_ADVERSARIAL}) | Qwen {N_QWEN}x(b={N_SAMPLES_BASE}, a={N_SAMPLES_ADVERSARIAL})")
with ThreadPoolExecutor(max_workers=5) as executor:
    futures = [executor.submit(call_llm_tracked, idx, row, method, model) for idx, row, method, model in jobs]
    for f in as_completed(futures):
        f.result()

df_new = pd.DataFrame(results)
for (method, model), group in df_new.groupby(['method', 'model']):
    path = get_path(method, model)
    print(f"Saved to {path}")
    print(pd.read_csv(path)[['ID', 'llm_prediction', 'solution', 'llm_escalate', 'method', 'model']].to_string())
