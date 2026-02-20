import os, re, datetime
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import openai

MODEL     = "gpt-5-nano"
METHOD    = "auditor"  # "base" or "auditor"
N_SAMPLES = 5

df = pd.read_csv("../data/AIME_Dataset_1983_2024.csv")

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

def call_llm(idx, row):
    base = create_prompt_base(row)
    if METHOD == "base":
        result = get_llm_base(base)
        return {**row, "prompt": base, "llm_prediction": result["pred"],
                "llm_delegate": result["del"], "solution": row["Answer"], "method": METHOD}
    elif METHOD == "auditor":
        result = get_sequential_inference(base)
        return {**row, "prompt": base, "llm_full_thought": result["full_thought"],
                "llm_prediction": result["pred"], "llm_critique": result["critique"],
                "llm_delegate": result["del"], "solution": row["Answer"], "method": METHOD}

sampled_rows = df.sample(n=N_SAMPLES)
results = []
completed = 0

def call_llm_tracked(idx, row):
    global completed
    result = call_llm(idx, row)
    completed += 1
    print(f"[{completed}/{N_SAMPLES}] Done: row {idx}")
    return result

print(f"Starting {N_SAMPLES} samples | model: {MODEL} | method: {METHOD}")
with ThreadPoolExecutor(max_workers=5) as executor:
    futures = [executor.submit(call_llm_tracked, idx, row) for idx, row in sampled_rows.iterrows()]
    for f in as_completed(futures):
        results.append(f.result())

df_results = pd.DataFrame(results)
df_results['timestamp'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

local_dir = '../results/AIME'
os.makedirs(local_dir, exist_ok=True)
local_path = os.path.join(local_dir, f'{METHOD}_{MODEL}.csv')

dropbox_dir = '/Users/mdisorbo/Harvard University Dropbox/Matthew DosSantos DiSorbo/Mac/Desktop/Trust/AIME'
os.makedirs(dropbox_dir, exist_ok=True)
dropbox_path = os.path.join(dropbox_dir, f'{METHOD}_{MODEL}.csv')

try:
    df_results = pd.concat([pd.read_csv(local_path), df_results], ignore_index=True)
except FileNotFoundError:
    pass

df_results.to_csv(local_path, index=False)
df_results.to_csv(dropbox_path, index=False)
print(f"Saved to {local_path}")
print(f"Saved to {dropbox_path}")
print(df_results[['ID', 'llm_prediction', 'solution', 'llm_delegate']].to_string())

os.system(f"cd .. && git add results/AIME/ && git commit -m 'Add AIME results' && git push")
