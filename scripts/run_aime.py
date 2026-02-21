import os, re, datetime, threading
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import openai

OAI_MODEL  = "gpt-5-nano"
QWEN_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

N_SAMPLES_BASE    = 1
N_SAMPLES_AUDITOR = 1
N_OAI  = 1
N_QWEN = 1

df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/AIME_Dataset_1983_2024.csv"))

# --- Qwen setup ---
qwen_pipe = None
qwen_lock = threading.Lock()

if N_QWEN > 0:
    from transformers import pipeline
    print(f"Loading {QWEN_MODEL}...")
    qwen_pipe = pipeline("text-generation", model=QWEN_MODEL, torch_dtype="auto", device_map="auto")
    print("Qwen loaded.")

def llm(prompt, model):
    if model == QWEN_MODEL:
        with qwen_lock:
            out = qwen_pipe([{"role": "user", "content": prompt}], max_new_tokens=512)
        return out[0]["generated_text"][-1]["content"]
    else:
        r = openai.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}])
        return r.choices[0].message.content.strip()

def create_prompt_base(row):
    return f"You are helping Person 1 solve the following math problem: {row['Question']}."

def get_llm_base(prompt, model):
    instructions = (
        "\n\nRespond with two numbers ONLY on two lines. "
        "First, your integer solution. "
        "Second, 0 to implement your solution or 1 to delegate to Person 1."
    )
    response = llm(prompt + instructions, model)
    lines = [l.strip() for l in response.strip().split('\n') if l.strip()]
    try:
        return {"pred": int(lines[0]), "del": int(lines[1])}
    except (ValueError, IndexError):
        print(f"Parse error: {response}")
        return {"pred": None, "del": None}

def get_sequential_inference(scenario, model):
    try:
        thought = llm(f"{scenario}\n\nSolve the math problem. Explain your reasoning in 1 sentence, "
                      "then conclude with 'SOLUTION: <integer>'.", model)
        preds = re.findall(r'SOLUTION:\s*(\d+)', thought)
        pred = preds[-1] if preds else "0"
        critique = llm(f"PROBLEM: {scenario}\n\nPROPOSED SOLUTION: {thought}\n\nCritique this reasoning in 1 sentence.", model)
        decision = llm(f"PROBLEM: {scenario}\nSOLUTION AND REASONING: {thought}\nCRITIQUE: {critique}\n\n"
                       "Implement your solution (0) or delegate to Person 1 (1). Respond with the digit ONLY.", model)
        return {"full_thought": thought, "pred": pred, "critique": critique, "del": decision}
    except Exception as e:
        return {"full_thought": str(e), "pred": "Err", "critique": "Err", "del": "1"}

def call_llm(idx, row, method, model):
    base = create_prompt_base(row)
    if method == "base":
        result = get_llm_base(base, model)
        return {**row, "prompt": base, "llm_prediction": result["pred"],
                "llm_delegate": result["del"], "solution": row["Answer"],
                "method": method, "model": model}
    elif method == "auditor":
        result = get_sequential_inference(base, model)
        return {**row, "prompt": base, "llm_full_thought": result["full_thought"],
                "llm_prediction": result["pred"], "llm_critique": result["critique"],
                "llm_delegate": result["del"], "solution": row["Answer"],
                "method": method, "model": model}

# --- Output ---
local_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../results/AIME")
os.makedirs(local_dir, exist_ok=True)
local_path = os.path.join(local_dir, "results.csv")

try:
    df_existing = pd.read_csv(local_path)
except FileNotFoundError:
    df_existing = pd.DataFrame()

results = []
completed = 0
total = (N_OAI + N_QWEN) * (N_SAMPLES_BASE + N_SAMPLES_AUDITOR)
save_lock = threading.Lock()

def save_progress():
    df_new = pd.DataFrame(results)
    df_new['timestamp'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pd.concat([df_existing, df_new], ignore_index=True).to_csv(local_path, index=False)

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
for model, n in [(OAI_MODEL, N_OAI), (QWEN_MODEL, N_QWEN)]:
    if n > 0:
        for method, n_samples in [("base", N_SAMPLES_BASE), ("auditor", N_SAMPLES_AUDITOR)]:
            if n_samples > 0:
                for idx, row in df.sample(n=n * n_samples).iterrows():
                    jobs.append((idx, row, method, model))

print(f"Starting {total} jobs | OAI {N_OAI}x(b={N_SAMPLES_BASE}, a={N_SAMPLES_AUDITOR}) | Qwen {N_QWEN}x(b={N_SAMPLES_BASE}, a={N_SAMPLES_AUDITOR})")
with ThreadPoolExecutor(max_workers=5) as executor:
    futures = [executor.submit(call_llm_tracked, idx, row, method, model) for idx, row, method, model in jobs]
    for f in as_completed(futures):
        f.result()

print(f"Saved to {local_path}")
df_final = pd.read_csv(local_path)
print(df_final[['ID', 'llm_prediction', 'solution', 'llm_delegate', 'method', 'model']].to_string())
