import os, re, sys, datetime, threading, random
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import openai

OAI_MODEL      = "gpt-5-mini-2025-08-07"
OAI_MODEL_NANO = "gpt-5-nano-2025-08-07"
QWEN_MODEL     = "Qwen/Qwen3.5-35B-A3B"
if len(sys.argv) > 1:
    QWEN_MODEL = sys.argv[1]

N_SAMPLES_BASE    = 50
N_SAMPLES_ADVERSARIAL = 50
N_OAI  = int(os.environ.get("N_OAI", 0))
N_NANO = int(os.environ.get("N_NANO", 0))
N_QWEN = int(os.environ.get("N_QWEN", 1))

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

def create_prompt_base(row):
    return (
        f"You are helping a group of crowd-workers label Wikipedia discussion comments as toxic or not. "
        f"This comment needs to be checked: '{row['comment']}'."
    )

def get_llm_base(base_prompt, model):
    # Step 1: Get prediction with reasoning
    pred_prompt = (
        f"{base_prompt}\n\n"
        "TASK: Predict the crowd-workers' aggregate decision. Explain your reasoning in 1 sentence. "
        "Then conclude with exactly 'PREDICTION: 1' (toxic) or 'PREDICTION: 0' (not toxic)."
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
        "Output 0 to implement your prediction, or 1 to escalate to the crowd-workers.\n"
        "A decision is correct if it equals the crowd-workers' true aggregate decision."
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
            "Output 0 to implement your prediction, or 1 to escalate to the crowd-workers.\n"
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
        trace = result.get("trace", f"[PROMPT]\n{result['full_prompt']}\n\n[RESPONSE]\n{result['response']}")
        return {**common, 'llm_prediction': result['pred'], 'llm_escalate': result['del'], 'trace': trace}
    elif method == "adversarial":
        result = get_sequential_inference(base, model)
        trace = (f"[PROMPT]\n{base}\n\n[THOUGHT]\n{result['full_thought']}\n\n"
                 f"[CRITIQUE]\n{result['critique']}\n\n[DECISION PROMPT]\n{result['decision_prompt']}\n\n[DECISION]\n{result['decision']}")
        return {**common, 'llm_prediction': result['pred'], 'llm_escalate': result['del'],
                'llm_full_thought': result['full_thought'], 'llm_critique': result['critique'], 'trace': trace}

# --- Output ---
local_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../results/WikipediaToxicity")
os.makedirs(local_dir, exist_ok=True)

def get_path(method, model):
    return os.path.join(local_dir, f'{method}_{model.split("/")[-1]}.csv')

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
for model, n in [(OAI_MODEL, N_OAI), (OAI_MODEL_NANO, N_NANO), (QWEN_MODEL, N_QWEN)]:
    if n > 0:
        for method, n_samples in [("base", N_SAMPLES_BASE), ("adversarial", N_SAMPLES_ADVERSARIAL)]:
            if n_samples > 0:
                sampled = random.sample(all_indices, n * n_samples)
                for idx in sampled:
                    jobs.append((idx, method, model))

print(f"Starting {total} jobs | OAI {N_OAI}x(b={N_SAMPLES_BASE}, a={N_SAMPLES_ADVERSARIAL}) | Nano {N_NANO}x(b={N_SAMPLES_BASE}, a={N_SAMPLES_ADVERSARIAL}) | Qwen {N_QWEN}x(b={N_SAMPLES_BASE}, a={N_SAMPLES_ADVERSARIAL})", flush=True)
with ThreadPoolExecutor(max_workers=5) as executor:
    futures = [executor.submit(call_llm_tracked, idx, method, model) for idx, method, model in jobs]
    for f in as_completed(futures):
        f.result()

df_new = pd.DataFrame([r for r in results if r is not None])
for (method, model), group in df_new.groupby(['method', 'model']):
    path = get_path(method, model)
    print(f"Saved to {path}", flush=True)
    print(pd.read_csv(path)[['rev_id', 'llm_prediction', 'human_response', 'llm_escalate', 'method', 'model']].to_string(), flush=True)
