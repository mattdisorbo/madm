import os, re, datetime, threading
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import openai

MODEL             = "gpt-5-nano"
N_SAMPLES_BASE    = 1
N_SAMPLES_AUDITOR = 1

df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/AIME_Dataset_1983_2024.csv"))

def create_prompt_base(row):
    return f"You are helping Person 1 solve the following math problem: {row['Question']}."

def llm(prompt):
    r = openai.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": prompt}])
    return r.choices[0].message.content.strip()

def get_llm_base(prompt):
    instructions = (
        "\n\nRespond with two numbers ONLY on two lines. "
        "First, your integer solution. "
        "Second, 0 to implement your solution or 1 to delegate to Person 1."
    )
    response = openai.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt + instructions}]
    )
    lines = [l.strip() for l in response.choices[0].message.content.strip().split('\n') if l.strip()]
    try:
        return {"pred": int(lines[0]), "del": int(lines[1])}
    except (ValueError, IndexError):
        print(f"Parse error: {response.choices[0].message.content}")
        return {"pred": None, "del": None}

def get_sequential_inference(scenario):
    try:
        thought = llm(f"{scenario}\n\nSolve the math problem. Explain your reasoning in 1 sentence, "
                      "then conclude with 'SOLUTION: <integer>'.")
        preds = re.findall(r'SOLUTION:\s*(\d+)', thought)
        pred = preds[-1] if preds else "0"
        critique = llm(f"PROBLEM: {scenario}\n\nPROPOSED SOLUTION: {thought}\n\nCritique this reasoning in 1 sentence.")
        decision = llm(f"PROBLEM: {scenario}\nSOLUTION AND REASONING: {thought}\nCRITIQUE: {critique}\n\n"
                       "Implement your solution (0) or delegate to Person 1 (1). Respond with the digit ONLY.")
        return {"full_thought": thought, "pred": pred, "critique": critique, "del": decision}
    except Exception as e:
        return {"full_thought": str(e), "pred": "Err", "critique": "Err", "del": "1"}

def call_llm(idx, row, method):
    base = create_prompt_base(row)
    if method == "base":
        result = get_llm_base(base)
        return {**row, "prompt": base, "llm_prediction": result["pred"],
                "llm_delegate": result["del"], "solution": row["Answer"],
                "method": method, "model": MODEL}
    elif method == "auditor":
        result = get_sequential_inference(base)
        return {**row, "prompt": base, "llm_full_thought": result["full_thought"],
                "llm_prediction": result["pred"], "llm_critique": result["critique"],
                "llm_delegate": result["del"], "solution": row["Answer"],
                "method": method, "model": MODEL}

local_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../results/AIME")
os.makedirs(local_dir, exist_ok=True)
local_path = os.path.join(local_dir, f'{MODEL}.csv')

try:
    df_existing = pd.read_csv(local_path)
except FileNotFoundError:
    df_existing = pd.DataFrame()

results = []
completed = 0
total = N_SAMPLES_BASE + N_SAMPLES_AUDITOR
save_lock = threading.Lock()

def save_progress():
    df_new = pd.DataFrame(results)
    df_new['timestamp'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pd.concat([df_existing, df_new], ignore_index=True).to_csv(local_path, index=False)

def call_llm_tracked(idx, row, method):
    global completed
    result = call_llm(idx, row, method)
    with save_lock:
        completed += 1
        results.append(result)
        print(f"[{completed}/{total}] Done: row {idx} ({method})")
        save_progress()
    return result

base_rows    = df.sample(n=N_SAMPLES_BASE)    if N_SAMPLES_BASE    > 0 else pd.DataFrame()
auditor_rows = df.sample(n=N_SAMPLES_AUDITOR) if N_SAMPLES_AUDITOR > 0 else pd.DataFrame()

print(f"Starting {N_SAMPLES_BASE} base + {N_SAMPLES_AUDITOR} auditor samples | model: {MODEL}")
with ThreadPoolExecutor(max_workers=5) as executor:
    futures = (
        [executor.submit(call_llm_tracked, idx, row, "base")    for idx, row in base_rows.iterrows()] +
        [executor.submit(call_llm_tracked, idx, row, "auditor") for idx, row in auditor_rows.iterrows()]
    )
    for f in as_completed(futures):
        f.result()

print(f"Saved to {local_path}")
df_final = pd.read_csv(local_path)
print(df_final[['ID', 'llm_prediction', 'solution', 'llm_delegate', 'method', 'model']].to_string())
