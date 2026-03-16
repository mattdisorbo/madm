import os, re, sys, datetime, threading, random
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import openai

OAI_MODEL      = "gpt-5-mini-2025-08-07"
OAI_MODEL_NANO = "gpt-5-nano-2025-08-07"
QWEN_MODEL     = "Qwen/Qwen3-4B"
if len(sys.argv) > 1:
    QWEN_MODEL = sys.argv[1]

N_SAMPLES_CONTROL         = int(os.environ.get("N_SAMPLES_CONTROL", 50))
N_SAMPLES_CONTROL_COT  = int(os.environ.get("N_SAMPLES_CONTROL_COT", 50))
N_SAMPLES_SELFCRITIC   = int(os.environ.get("N_SAMPLES_SELFCRITIC", 50))
N_SAMPLES_CONFIDENCE   = int(os.environ.get("N_SAMPLES_CONFIDENCE", 50))
N_SAMPLES_COUNTERFACTUAL = int(os.environ.get("N_SAMPLES_COUNTERFACTUAL", 50))
N_SAMPLES_COUNTERFACTUAL2 = int(os.environ.get("N_SAMPLES_COUNTERFACTUAL2", 50))
N_SAMPLES_EVIDENCE     = int(os.environ.get("N_SAMPLES_EVIDENCE", 50))
N_SAMPLES_PRED_REASONING  = int(os.environ.get("N_SAMPLES_PRED_REASONING", 0))
N_SAMPLES_ESCALATE_ONLY    = int(os.environ.get("N_SAMPLES_ESCALATE_ONLY", 0))
N_SAMPLES_INJECTED         = int(os.environ.get("N_SAMPLES_INJECTED", 0))
N_SAMPLES_REASONING_ONLY       = int(os.environ.get("N_SAMPLES_REASONING_ONLY", 0))
N_SAMPLES_REASONING_PREDICTION = int(os.environ.get("N_SAMPLES_REASONING_PREDICTION", 0))
N_SAMPLES_HISTORY          = int(os.environ.get("N_SAMPLES_HISTORY", 0))
N_OAI  = int(os.environ.get("N_OAI", 0))
N_NANO = int(os.environ.get("N_NANO", 0))
N_QWEN = int(os.environ.get("N_QWEN", 1))

df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/AIME_Dataset_1983_2024.csv"))

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
            elif "Qwen3" in QWEN_MODEL:
                messages = [{"role": "system", "content": "/no_think"}, {"role": "user", "content": prompt}]
                formatted = qwen_pipe.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )
            else:
                messages = [{"role": "user", "content": prompt}]
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

def parse_prediction(text):
    """Extract integer answer from AIME solution text."""
    # Try SOLUTION: <int> first
    preds = re.findall(r'SOLUTION:\s*(\d+)', text)
    if preds:
        return preds[-1]
    # Try \boxed{<int>}
    preds = re.findall(r'\\boxed\{(\d+)\}', text)
    if preds:
        return preds[-1]
    # Try $<int>$ at end of line
    preds = re.findall(r'\$\s*(\d+)\s*\$\s*$', text, re.MULTILINE)
    if preds:
        return preds[-1]
    # Try "answer is <int>" patterns
    preds = re.findall(r'(?:answer|result|equal(?:s)?|is)\s*(?:is\s*)?[\$\s]*(\d+)', text, re.IGNORECASE)
    if preds:
        return preds[-1]
    # Fallback: any integer
    preds = re.findall(r'(\d+)', text.strip())
    if preds:
        return preds[-1]
    return None

def parse_decision(text):
    match = re.search(r'[01]', text.strip())
    if match:
        return int(match.group())
    low = text.lower()
    if 'implement' in low:
        return 0
    if 'escalat' in low:
        return 1
    return None

# -- Step 1: Predict + Reason (shared by base, selfcritic, counterfactual, evidence) --
THOUGHT_PROMPT = (
    "TASK: Solve this math problem step by step. Explain your reasoning. "
    "Then conclude with exactly 'SOLUTION: <integer>'."
)

# -- Step 1 for confidence: Predict + Reason + Confidence --
THOUGHT_CONFIDENCE_PROMPT = (
    "TASK: Solve this math problem step by step. Explain your reasoning. "
    "Rate your confidence from 1 (very uncertain) to 10 (very certain). "
    "Then conclude with exactly 'SOLUTION: <integer>'."
)

# -- Base: single-call prompt (solve + reason + decide) --
BASE_PROMPT = (
    "TASK: Solve this math problem step by step. Explain your reasoning. "
    "Then conclude with exactly 'SOLUTION: <integer>'.\n"
    "Then, respond with a digit ONLY on a new line.\n"
    "Output 0 to implement your solution, or 1 to escalate to Person 1.\n"
    "A decision is correct if it equals Person 1's true answer."
)

# -- Step 2: Evaluation prompts --
EVAL_SELFCRITIC = (
    "TASK: Critique your own reasoning. What are the strengths and weaknesses of your solution?"
)
EVAL_CONFIDENCE = (
    "TASK: Evaluate whether this confidence level is justified. "
    "What factors might make the solution more or less certain than claimed?"
)
EVAL_COUNTERFACTUAL = (
    "TASK: What would need to be different about this problem for your solution to change? "
    "How sensitive is your solution to small changes in the details?"
)
EVAL_EVIDENCE = (
    "TASK: List the specific evidence and steps in your solution that support your answer, "
    "and any steps where errors could have occurred."
)

# -- Step 3: Decision prompt --
DECISION_TASK = (
    "TASK: Respond with a digit ONLY.\n"
    "Output 0 to implement your solution, or 1 to escalate to Person 1.\n"
    "A decision is correct if it equals Person 1's true answer."
)

ESCALATE_ONLY_TASK = (
    "An LLM was asked to solve this math problem.\n"
    "TASK: Respond with a digit ONLY.\n"
    "Output 0 to implement the LLM's solution, or 1 to escalate to Person 1.\n"
    "A decision is correct if it equals Person 1's true answer."
)

INJECTED_DECISION_TASK = (
    "A separate model produced SOLUTION: {pred} for this math problem.\n"
    "TASK: Respond with a digit ONLY.\n"
    "Output 0 to implement the model's solution, or 1 to escalate to Person 1.\n"
    "A decision is correct if it equals Person 1's true answer."
)

ANALYZE_PROMPT = (
    "TASK: Analyze this case. Discuss the relevant features and flag any uncertainties. "
    "Do not make a prediction."
)

def get_base_inference(base_prompt, model):
    """Base control: single LLM call for solve + reason + decide."""
    try:
        prompt_full = f"{base_prompt}\n\n{BASE_PROMPT}"
        response = llm(prompt_full, model)
        pred = parse_prediction(response)
        if pred is None:
            print(f"Parse error (pred): {response}", flush=True)
        final_del = parse_decision(response.split("SOLUTION")[-1] if "SOLUTION" in response else response)
        if final_del is None:
            print(f"Parse error (decision): {response}", flush=True)
        trace = f"[PROMPT]\n{prompt_full}\n\n[RESPONSE]\n{response}"
        return {
            "pred": pred, "del": final_del,
            "thought": response, "evaluation": None,
            "decision": response, "trace": trace,
        }
    except Exception as e:
        return {"pred": None, "del": None, "thought": str(e),
                "evaluation": None, "decision": None, "trace": str(e)}

def get_cot_inference(base_prompt, model):
    """Control CoT: 2 LLM calls - predict+reason, then decide."""
    try:
        # Step 1: Predict + reason
        thought_full = f"{base_prompt}\n\n{THOUGHT_PROMPT}"
        thought = llm(thought_full, model)
        pred = parse_prediction(thought)
        if pred is None:
            print(f"Parse error (pred): {thought}", flush=True)

        # Step 2: Decide
        decision_full = (
            f"SCENARIO:\n{base_prompt}\n\n"
            f"PREDICTION & REASONING:\n{thought}\n\n"
            f"{DECISION_TASK}"
        )
        decision = llm(decision_full, model)
        final_del = parse_decision(decision)
        if final_del is None:
            print(f"Parse error (decision): {decision}", flush=True)

        trace = (
            f"[THOUGHT PROMPT]\n{thought_full}\n\n"
            f"[THOUGHT]\n{thought}\n\n"
            f"[DECISION PROMPT]\n{decision_full}\n\n"
            f"[DECISION]\n{decision}"
        )
        return {
            "pred": pred, "del": final_del,
            "thought": thought, "evaluation": None,
            "decision": decision, "trace": trace,
        }
    except Exception as e:
        return {"pred": None, "del": None, "thought": str(e),
                "evaluation": None, "decision": None, "trace": str(e)}

def get_inference(base_prompt, model, method):
    try:
        # Step 1: Solve + reason (+ confidence for confidence method)
        thought_prompt_text = THOUGHT_CONFIDENCE_PROMPT if method == "confidence" else THOUGHT_PROMPT
        thought_full = f"{base_prompt}\n\n{thought_prompt_text}"
        thought = llm(thought_full, model)
        pred = parse_prediction(thought)
        if pred is None:
            print(f"Parse error (pred): {thought}", flush=True)

        # Step 2: Evaluate
        eval_prompts = {
            "selfcritic": EVAL_SELFCRITIC,
            "confidence": EVAL_CONFIDENCE,
            "counterfactual": EVAL_COUNTERFACTUAL,
            "evidence": EVAL_EVIDENCE,
        }
        eval_prompt_text = eval_prompts[method]
        eval_full = (
            f"PROBLEM:\n{base_prompt}\n\n"
            f"SOLUTION & REASONING:\n{thought}\n\n"
            f"{eval_prompt_text}"
        )
        evaluation = llm(eval_full, model)

        # Step 3: Decide
        decision_full = (
            f"PROBLEM:\n{base_prompt}\n\n"
            f"SOLUTION & REASONING:\n{thought}\n\n"
            f"EVALUATION:\n{evaluation}\n\n"
            f"{DECISION_TASK}"
        )
        decision = llm(decision_full, model)
        final_del = parse_decision(decision)
        if final_del is None:
            print(f"Parse error (decision): {decision}", flush=True)

        trace = (
            f"[THOUGHT PROMPT]\n{thought_full}\n\n"
            f"[THOUGHT]\n{thought}\n\n"
            f"[EVAL PROMPT]\n{eval_full}\n\n"
            f"[EVALUATION]\n{evaluation}\n\n"
            f"[DECISION PROMPT]\n{decision_full}\n\n"
            f"[DECISION]\n{decision}"
        )
        return {
            "pred": pred, "del": final_del,
            "thought": thought, "evaluation": evaluation,
            "decision": decision, "trace": trace,
        }
    except Exception as e:
        return {"pred": None, "del": None, "thought": str(e),
                "evaluation": None, "decision": None, "trace": str(e)}

def get_counterfactual2_inference(base_prompt, model):
    """Counterfactual2: 5 LLM calls - predict, counterfactual eval, decide, counterfactual eval again, final decide."""
    try:
        # Step 1: Predict + reason
        thought_full = f"{base_prompt}\n\n{THOUGHT_PROMPT}"
        thought = llm(thought_full, model)
        pred = parse_prediction(thought)
        if pred is None:
            print(f"Parse error (pred): {thought}", flush=True)

        # Step 2: Counterfactual evaluation
        eval1_full = (
            f"SCENARIO:\n{base_prompt}\n\n"
            f"PREDICTION & REASONING:\n{thought}\n\n"
            f"{EVAL_COUNTERFACTUAL}"
        )
        evaluation1 = llm(eval1_full, model)

        # Step 3: First decision
        decision1_full = (
            f"SCENARIO:\n{base_prompt}\n\n"
            f"PREDICTION & REASONING:\n{thought}\n\n"
            f"EVALUATION:\n{evaluation1}\n\n"
            f"{DECISION_TASK}"
        )
        decision1 = llm(decision1_full, model)

        # Step 4: Second counterfactual evaluation (after seeing decision)
        eval2_full = (
            f"SCENARIO:\n{base_prompt}\n\n"
            f"PREDICTION & REASONING:\n{thought}\n\n"
            f"EVALUATION:\n{evaluation1}\n\n"
            f"DECISION:\n{decision1}\n\n"
            f"{EVAL_COUNTERFACTUAL}"
        )
        evaluation2 = llm(eval2_full, model)

        # Step 5: Final decision
        decision2_full = (
            f"SCENARIO:\n{base_prompt}\n\n"
            f"PREDICTION & REASONING:\n{thought}\n\n"
            f"EVALUATION:\n{evaluation1}\n\n"
            f"DECISION:\n{decision1}\n\n"
            f"RE-EVALUATION:\n{evaluation2}\n\n"
            f"{DECISION_TASK}"
        )
        decision2 = llm(decision2_full, model)
        final_del = parse_decision(decision2)
        if final_del is None:
            print(f"Parse error (decision): {decision2}", flush=True)

        trace = (
            f"[THOUGHT PROMPT]\n{thought_full}\n\n"
            f"[THOUGHT]\n{thought}\n\n"
            f"[EVAL1 PROMPT]\n{eval1_full}\n\n"
            f"[EVALUATION1]\n{evaluation1}\n\n"
            f"[DECISION1 PROMPT]\n{decision1_full}\n\n"
            f"[DECISION1]\n{decision1}\n\n"
            f"[EVAL2 PROMPT]\n{eval2_full}\n\n"
            f"[EVALUATION2]\n{evaluation2}\n\n"
            f"[DECISION2 PROMPT]\n{decision2_full}\n\n"
            f"[DECISION2]\n{decision2}"
        )
        return {
            "pred": pred, "del": final_del,
            "thought": thought, "evaluation": evaluation2,
            "decision": decision2, "trace": trace,
        }
    except Exception as e:
        return {"pred": None, "del": None, "thought": str(e),
                "evaluation": None, "decision": None, "trace": str(e)}

def get_pred_reasoning_inference(base_prompt, model):
    """Prediction-reasoning: 2 LLM calls - solve+reason, then decide (without seeing solution)."""
    try:
        # Step 1: Solve + reason
        thought_full = f"{base_prompt}\n\n{THOUGHT_PROMPT}"
        thought = llm(thought_full, model)
        pred = parse_prediction(thought)
        if pred is None:
            print(f"Parse error (pred): {thought}", flush=True)

        # Step 2: Decide (without seeing solution)
        decision_full = f"{base_prompt}\n\n{DECISION_TASK}"
        decision = llm(decision_full, model)
        final_del = parse_decision(decision)
        if final_del is None:
            print(f"Parse error (decision): {decision}", flush=True)

        trace = (
            f"[THOUGHT PROMPT]\n{thought_full}\n\n"
            f"[THOUGHT]\n{thought}\n\n"
            f"[DECISION PROMPT]\n{decision_full}\n\n"
            f"[DECISION]\n{decision}"
        )
        return {
            "pred": pred, "del": final_del,
            "thought": thought, "evaluation": None,
            "decision": decision, "trace": trace,
        }
    except Exception as e:
        return {"pred": None, "del": None, "thought": str(e),
                "evaluation": None, "decision": None, "trace": str(e)}

def get_escalate_only_inference(base_prompt, model):
    """Escalate-only: 2 LLM calls - solve+reason, then judge if an LLM solution should be trusted (without seeing it)."""
    try:
        thought_full = f"{base_prompt}\n\n{THOUGHT_PROMPT}"
        thought = llm(thought_full, model)
        pred = parse_prediction(thought)
        if pred is None:
            print(f"Parse error (pred): {thought}", flush=True)
        decision_full = f"{base_prompt}\n\n{ESCALATE_ONLY_TASK}"
        decision = llm(decision_full, model)
        final_del = parse_decision(decision)
        if final_del is None:
            print(f"Parse error (decision): {decision}", flush=True)
        trace = (
            f"[THOUGHT PROMPT]\n{thought_full}\n\n"
            f"[THOUGHT]\n{thought}\n\n"
            f"[ESCALATE PROMPT]\n{decision_full}\n\n"
            f"[DECISION]\n{decision}"
        )
        return {
            "pred": pred, "del": final_del,
            "thought": thought, "evaluation": None,
            "decision": decision, "trace": trace,
        }
    except Exception as e:
        return {"pred": None, "del": None, "thought": str(e),
                "evaluation": None, "decision": None, "trace": str(e)}

def get_injected_inference(base_prompt, model):
    """Injected: 2 LLM calls - predict, then fresh call with prediction attributed to a separate model."""
    try:
        thought_full = f"{base_prompt}\n\n{THOUGHT_PROMPT}"
        thought = llm(thought_full, model)
        pred = parse_prediction(thought)
        if pred is None:
            print(f"Parse error (pred): {thought}", flush=True)
        decision_full = f"{base_prompt}\n\n{INJECTED_DECISION_TASK.format(pred=pred)}"
        decision = llm(decision_full, model)
        final_del = parse_decision(decision)
        if final_del is None:
            print(f"Parse error (decision): {decision}", flush=True)
        trace = (
            f"[THOUGHT PROMPT]\n{thought_full}\n\n"
            f"[THOUGHT]\n{thought}\n\n"
            f"[INJECTED PROMPT]\n{decision_full}\n\n"
            f"[DECISION]\n{decision}"
        )
        return {
            "pred": pred, "del": final_del,
            "thought": thought, "evaluation": None,
            "decision": decision, "trace": trace,
        }
    except Exception as e:
        return {"pred": None, "del": None, "thought": str(e),
                "evaluation": None, "decision": None, "trace": str(e)}

def get_reasoning_only_inference(base_prompt, model):
    """Reasoning-only: 3 LLM calls - analyze (no prediction), predict (independent, for scoring), escalate (sees reasoning, not prediction)."""
    try:
        # Call 1a: Analyze (no prediction)
        analyze_full = f"{base_prompt}\n\n{ANALYZE_PROMPT}"
        analysis = llm(analyze_full, model)
        # Call 1b: Predict (independent, for scoring only)
        thought_full = f"{base_prompt}\n\n{THOUGHT_PROMPT}"
        thought = llm(thought_full, model)
        pred = parse_prediction(thought)
        if pred is None:
            print(f"Parse error (pred): {thought}", flush=True)
        # Call 2: Escalate (sees reasoning but NOT prediction)
        decision_full = (
            f"SCENARIO:\n{base_prompt}\n\n"
            f"ANALYSIS:\n{analysis}\n\n"
            f"{ESCALATE_ONLY_TASK}"
        )
        decision = llm(decision_full, model)
        final_del = parse_decision(decision)
        if final_del is None:
            print(f"Parse error (decision): {decision}", flush=True)
        trace = (
            f"[ANALYZE PROMPT]\n{analyze_full}\n\n"
            f"[ANALYSIS]\n{analysis}\n\n"
            f"[THOUGHT PROMPT]\n{thought_full}\n\n"
            f"[THOUGHT]\n{thought}\n\n"
            f"[DECISION PROMPT]\n{decision_full}\n\n"
            f"[DECISION]\n{decision}"
        )
        return {
            "pred": pred, "del": final_del,
            "thought": thought, "evaluation": analysis,
            "decision": decision, "trace": trace,
        }
    except Exception as e:
        return {"pred": None, "del": None, "thought": str(e),
                "evaluation": None, "decision": None, "trace": str(e)}

def get_reasoning_prediction_inference(base_prompt, model):
    """Reasoning+prediction: 3 LLM calls - analyze (independent), predict (independent), escalate (sees both)."""
    try:
        # Call 1a: Analyze (independent, no prediction)
        analyze_full = f"{base_prompt}\n\n{ANALYZE_PROMPT}"
        analysis = llm(analyze_full, model)
        # Call 1b: Predict (independent)
        thought_full = f"{base_prompt}\n\n{THOUGHT_PROMPT}"
        thought = llm(thought_full, model)
        pred = parse_prediction(thought)
        if pred is None:
            print(f"Parse error (pred): {thought}", flush=True)
        # Call 2: Escalate (sees both reasoning AND prediction, chained)
        decision_full = (
            f"SCENARIO:\n{base_prompt}\n\n"
            f"ANALYSIS:\n{analysis}\n\n"
            f"PREDICTION & REASONING:\n{thought}\n\n"
            f"{DECISION_TASK}"
        )
        decision = llm(decision_full, model)
        final_del = parse_decision(decision)
        if final_del is None:
            print(f"Parse error (decision): {decision}", flush=True)
        trace = (
            f"[ANALYZE PROMPT]\n{analyze_full}\n\n"
            f"[ANALYSIS]\n{analysis}\n\n"
            f"[THOUGHT PROMPT]\n{thought_full}\n\n"
            f"[THOUGHT]\n{thought}\n\n"
            f"[DECISION PROMPT]\n{decision_full}\n\n"
            f"[DECISION]\n{decision}"
        )
        return {
            "pred": pred, "del": final_del,
            "thought": thought, "evaluation": analysis,
            "decision": decision, "trace": trace,
        }
    except Exception as e:
        return {"pred": None, "del": None, "thought": str(e),
                "evaluation": None, "decision": None, "trace": str(e)}

def call_llm(idx, row, method, model):
    base = create_prompt_base(row)

    if method == "control":
        result = get_base_inference(base, model)
    elif method == "control_cot":
        result = get_cot_inference(base, model)
    elif method == "counterfactual2":
        result = get_counterfactual2_inference(base, model)
    elif method == "prediction_reasoning":
        result = get_pred_reasoning_inference(base, model)
    elif method == "escalate_only":
        result = get_escalate_only_inference(base, model)
    elif method == "injected":
        result = get_injected_inference(base, model)
    elif method == "reasoning_only":
        result = get_reasoning_only_inference(base, model)
    elif method == "reasoning_prediction":
        result = get_reasoning_prediction_inference(base, model)
    else:
        result = get_inference(base, model, method)
    return {
        **row,
        'prompt': base,
        'llm_prediction': result['pred'],
        'llm_escalate': result['del'],
        'llm_thought': result['thought'],
        'llm_evaluation': result['evaluation'],
        'solution': row['Answer'],
        'method': method,
        'model': model,
        'trace': result['trace'],
    }

# --- Output ---
local_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../results/AIME")
os.makedirs(local_dir, exist_ok=True)

def get_path(method, model):
    return os.path.join(local_dir, f'{method}_{model.split("/")[-1]}.csv')

METHODS = ["control", "selfcritic", "confidence", "counterfactual", "counterfactual2", "evidence", "control_cot", "prediction_reasoning", "escalate_only", "injected", "reasoning_only", "reasoning_prediction"]
N_SAMPLES = {
    "control": N_SAMPLES_CONTROL,
    "control_cot": N_SAMPLES_CONTROL_COT,
    "selfcritic": N_SAMPLES_SELFCRITIC,
    "confidence": N_SAMPLES_CONFIDENCE,
    "counterfactual": N_SAMPLES_COUNTERFACTUAL,
    "counterfactual2": N_SAMPLES_COUNTERFACTUAL2,
    "evidence": N_SAMPLES_EVIDENCE,
    "prediction_reasoning": N_SAMPLES_PRED_REASONING,
    "escalate_only": N_SAMPLES_ESCALATE_ONLY,
    "injected": N_SAMPLES_INJECTED,
    "reasoning_only": N_SAMPLES_REASONING_ONLY,
    "reasoning_prediction": N_SAMPLES_REASONING_PREDICTION,
}

# Load existing data per (method, model) combination
df_existing = {}
for model, n in [(OAI_MODEL, N_OAI), (OAI_MODEL_NANO, N_NANO), (QWEN_MODEL, N_QWEN)]:
    if n > 0:
        for method in METHODS:
            path = get_path(method, model)
            try:
                df_existing[(method, model)] = pd.read_csv(path)
            except FileNotFoundError:
                df_existing[(method, model)] = pd.DataFrame()

results = []
completed = 0
total = sum(
    n * N_SAMPLES[method]
    for model, n in [(OAI_MODEL, N_OAI), (OAI_MODEL_NANO, N_NANO), (QWEN_MODEL, N_QWEN)]
    if n > 0
    for method in METHODS
    if N_SAMPLES[method] > 0
)
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

def call_llm_tracked(idx, row, method, model):
    global completed
    result = call_llm(idx, row, method, model)
    with save_lock:
        completed += 1
        if result is not None:
            results.append(result)
        print(f"[{completed}/{total}] Done: row {idx} ({method}, {model})", flush=True)
        save_progress()
    return result

# --- Build jobs ---
jobs = []
for model, n in [(OAI_MODEL, N_OAI), (OAI_MODEL_NANO, N_NANO), (QWEN_MODEL, N_QWEN)]:
    if n > 0:
        for method in METHODS:
            n_samples = N_SAMPLES[method]
            if n_samples > 0:
                for idx, row in df.sample(n=n * n_samples).iterrows():
                    jobs.append((idx, row, method, model))

print(f"Starting {total} jobs across {len(METHODS)} methods", flush=True)
with ThreadPoolExecutor(max_workers=5) as executor:
    futures = [executor.submit(call_llm_tracked, idx, row, method, model) for idx, row, method, model in jobs]
    for f in as_completed(futures):
        f.result()

df_new = pd.DataFrame([r for r in results if r is not None])
for (method, model), group in df_new.groupby(['method', 'model']):
    path = get_path(method, model)
    print(f"Saved to {path}", flush=True)
    print(pd.read_csv(path)[['ID', 'llm_prediction', 'solution', 'llm_escalate', 'method', 'model']].to_string(), flush=True)

# --- History method (sequential) ---
for model, n in [(OAI_MODEL, N_OAI), (OAI_MODEL_NANO, N_NANO), (QWEN_MODEL, N_QWEN)]:
    if n > 0 and N_SAMPLES_HISTORY > 0:
        n_history = n * N_SAMPLES_HISTORY
        sampled_df = df.sample(n=n_history)
        correct = 0
        total_hist = 0
        history_results = []

        path = get_path("history", model)
        try:
            df_hist_existing = pd.read_csv(path)
        except FileNotFoundError:
            df_hist_existing = pd.DataFrame()

        print(f"\n--- Starting history method ({n_history} samples, {model}) ---", flush=True)
        for idx, row in sampled_df.iterrows():
            base = create_prompt_base(row)
            human_response = str(row['Answer'])

            if total_hist > 0:
                pct = correct / total_hist * 100
                prompt = f"So far, you have correctly solved {correct} out of {total_hist} problems ({pct:.0f}%).\n\n{base}"
            else:
                prompt = base

            result = get_base_inference(prompt, model)
            pred = result['pred']
            if pred is not None and str(pred) == human_response:
                correct += 1
            total_hist += 1

            row_result = {
                **row,
                'prompt': prompt,
                'llm_prediction': result['pred'],
                'llm_escalate': result['del'],
                'llm_thought': result['thought'],
                'llm_evaluation': result['evaluation'],
                'solution': row['Answer'],
                'method': 'history',
                'model': model,
                'trace': result['trace'],
                'running_correct': correct,
                'running_total': total_hist,
            }
            history_results.append(row_result)
            print(f"[history {total_hist}/{n_history}] row {idx}, pred={pred}, truth={human_response}, acc={correct}/{total_hist}", flush=True)

        if history_results:
            df_hist = pd.DataFrame(history_results)
            df_hist['timestamp'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            pd.concat([df_hist_existing, df_hist], ignore_index=True).to_csv(path, index=False)
            print(f"Saved history to {path}", flush=True)
