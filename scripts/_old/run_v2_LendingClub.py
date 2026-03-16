import os, re, sys, datetime, threading, random
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
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
N_SAMPLES_CONTROL_RF      = int(os.environ.get("N_SAMPLES_CONTROL_RF", 50))
N_SAMPLES_CONTROL_COT_RF  = int(os.environ.get("N_SAMPLES_CONTROL_COT_RF", 50))
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

ACCEPTED_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/accepted_10k.csv")
REJECTED_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/rejected_10k.csv")

# --- Load and harmonize data ---
print("Loading LendingClub data...", flush=True)
acc = pd.read_csv(ACCEPTED_PATH)
rej = pd.read_csv(REJECTED_PATH)

acc_norm = pd.DataFrame({
    'loan_amnt': acc['loan_amnt'],
    'purpose':   acc['purpose'],
    'emp_length': acc['emp_length'],
    'dti':       acc['dti'],
    'fico':      acc['fico_range_low'],
    'accepted':  1,
})

rej_dti = rej['Debt-To-Income Ratio'].astype(str).str.replace('%', '', regex=False)
rej_norm = pd.DataFrame({
    'loan_amnt': rej['Amount Requested'],
    'purpose':   rej['Loan Title'],
    'emp_length': rej['Employment Length'],
    'dti':       pd.to_numeric(rej_dti, errors='coerce'),
    'fico':      pd.to_numeric(rej['Risk_Score'], errors='coerce'),
    'accepted':  0,
})

data = pd.concat([acc_norm, rej_norm], ignore_index=True)
data = data.dropna(subset=['loan_amnt', 'dti', 'fico']).reset_index(drop=True)

print(f"Loaded {len(data)} applications. Approval rate: {data['accepted'].mean():.3f}", flush=True)

# --- Holdout split ---
features = ['loan_amnt', 'dti', 'fico']
target = 'accepted'
df_clean = data[features + [target]].copy()
train_idx, holdout_idx = train_test_split(df_clean.index, test_size=0.2, random_state=42)
holdout_indices = list(holdout_idx)
print(f"Holdout size: {len(holdout_indices)}", flush=True)

# --- RF baseline ---
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(df_clean.loc[train_idx, features], df_clean.loc[train_idx, target])
rf_preds = rf_model.predict(df_clean[features])
rf_probs = rf_model.predict_proba(df_clean[features])
rf_accuracy = rf_model.score(df_clean.loc[holdout_idx, features], df_clean.loc[holdout_idx, target])
print(f"RF baseline accuracy: {rf_accuracy:.3f}", flush=True)

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
    return (
        f"You are helping Person 1, a loan officer, decide whether to approve a loan application. "
        f"The applicant is requesting ${row['loan_amnt']:,.0f} for {row['purpose']}. "
        f"They have been employed for {row['emp_length']}, "
        f"a debt-to-income ratio of {row['dti']:.1f}%, "
        f"and a credit score of {row['fico']:.0f}."
    )

def create_prompt_rf(row):
    base = create_prompt_base(row)
    idx = row.name if hasattr(row, 'name') else None
    if idx is not None and idx in df_clean.index:
        loc = df_clean.index.get_loc(idx)
        pred = rf_preds[loc]
        prob = rf_probs[loc]
        confidence = max(prob) * 100
        outcome = "approve" if pred == 1 else "reject"
        base += f" A statistical model predicts Person 1 will {outcome} this application ({confidence:.0f}% confidence)."
    return base

def parse_prediction(text):
    preds = re.findall(r'PREDICTION:\s*([01])', text)
    if preds:
        return int(preds[-1])
    digits = re.findall(r'[01]', text.strip())
    if digits:
        return int(digits[0])
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

# ── Step 1: Predict + Reason (shared by base, selfcritic, counterfactual, evidence) ──
THOUGHT_PROMPT = (
    "TASK: Predict Person 1's decision. Explain your reasoning in 1 sentence. "
    "Then conclude with exactly 'PREDICTION: 1' (approve) or 'PREDICTION: 0' (reject)."
)

# ── Step 1 for confidence: Predict + Reason + Confidence ──
THOUGHT_CONFIDENCE_PROMPT = (
    "TASK: Predict Person 1's decision. Explain your reasoning in 1 sentence. "
    "Rate your confidence from 1 (very uncertain) to 10 (very certain). "
    "Then conclude with exactly 'PREDICTION: 1' (approve) or 'PREDICTION: 0' (reject)."
)

# ── Base: single-call prompt (predict + reason + decide) ──
BASE_PROMPT = (
    "TASK: Predict Person 1's decision. Explain your reasoning in 1 sentence. "
    "Then conclude with exactly 'PREDICTION: 1' (approve) or 'PREDICTION: 0' (reject).\n"
    "Then, respond with a digit ONLY on a new line.\n"
    "Output 0 to implement your prediction, or 1 to escalate to Person 1.\n"
    "A decision is considered correct if it equals Person 1's true decision."
)

# ── Step 2: Evaluation prompts ──
EVAL_SELFCRITIC = (
    "TASK: Critique your own reasoning. What are the strengths and weaknesses of your prediction?"
)
EVAL_CONFIDENCE = (
    "TASK: Evaluate whether this confidence level is justified. "
    "What factors might make the prediction more or less certain than claimed?"
)
EVAL_COUNTERFACTUAL = (
    "TASK: What would need to be different about this scenario for your prediction to change? "
    "How sensitive is your prediction to small changes in the details?"
)
EVAL_EVIDENCE = (
    "TASK: List the specific evidence in this scenario that supports your prediction, "
    "and the evidence that contradicts it."
)

# ── Step 3: Decision prompt ──
DECISION_TASK = (
    "TASK: Respond with a digit ONLY.\n"
    "Output 0 to implement your prediction, or 1 to escalate to Person 1.\n"
    "A decision is considered correct if it equals Person 1's true decision."
)

ESCALATE_ONLY_TASK = (
    "An LLM was asked to predict Person 1's decision in this scenario.\n"
    "TASK: Respond with a digit ONLY.\n"
    "Output 0 to implement the LLM's prediction, or 1 to escalate to Person 1.\n"
    "A decision is considered correct if it equals Person 1's true decision."
)

INJECTED_DECISION_TASK = (
    "A separate model predicted PREDICTION: {pred} for Person 1's decision in this scenario.\n"
    "TASK: Respond with a digit ONLY.\n"
    "Output 0 to implement the model's prediction, or 1 to escalate to Person 1.\n"
    "A decision is considered correct if it equals Person 1's true decision."
)

ANALYZE_PROMPT = (
    "TASK: Analyze this case. Discuss the relevant features and flag any uncertainties. "
    "Do not make a prediction."
)

def get_base_inference(base_prompt, model):
    """Base control: single LLM call for predict + reason + decide."""
    try:
        prompt_full = f"{base_prompt}\n\n{BASE_PROMPT}"
        response = llm(prompt_full, model)
        pred = parse_prediction(response)
        if pred is None:
            print(f"Parse error (pred): {response}", flush=True)
        final_del = parse_decision(response.split("PREDICTION")[-1] if "PREDICTION" in response else response)
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
        # Step 1: Predict + reason (+ confidence for confidence method)
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
            f"SCENARIO:\n{base_prompt}\n\n"
            f"PREDICTION & REASONING:\n{thought}\n\n"
            f"{eval_prompt_text}"
        )
        evaluation = llm(eval_full, model)

        # Step 3: Decide
        decision_full = (
            f"SCENARIO:\n{base_prompt}\n\n"
            f"PREDICTION & REASONING:\n{thought}\n\n"
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
    """Prediction-reasoning: 2 LLM calls - predict+reason, then decide (without seeing prediction)."""
    try:
        # Step 1: Predict + reason
        thought_full = f"{base_prompt}\n\n{THOUGHT_PROMPT}"
        thought = llm(thought_full, model)
        pred = parse_prediction(thought)
        if pred is None:
            print(f"Parse error (pred): {thought}", flush=True)

        # Step 2: Decide (without seeing prediction)
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
    """Escalate-only: 2 LLM calls - predict+reason, then judge if an LLM prediction should be trusted (without seeing it)."""
    try:
        # Step 1: Predict + reason (just to get prediction)
        thought_full = f"{base_prompt}\n\n{THOUGHT_PROMPT}"
        thought = llm(thought_full, model)
        pred = parse_prediction(thought)
        if pred is None:
            print(f"Parse error (pred): {thought}", flush=True)

        # Step 2: Judge whether an LLM's prediction should be implemented (without seeing it)
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

def call_llm(row_idx, method, model):
    row = data.loc[row_idx]
    base = create_prompt_base(row)
    human_response = int(row['accepted'])

    common = {
        'index': row_idx,
        'loan_amnt': row['loan_amnt'],
        'purpose': row['purpose'],
        'human_response': human_response,
        'prompt': base,
        'method': method,
        'model': model,
    }

    if method in ("control", "control_rf"):
        prompt = create_prompt_rf(row) if method == "control_rf" else base
        result = get_base_inference(prompt, model)
    elif method in ("control_cot", "control_cot_rf"):
        prompt = create_prompt_rf(row) if method == "control_cot_rf" else base
        result = get_cot_inference(prompt, model)
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
        **common,
        'llm_prediction': result['pred'],
        'llm_escalate': result['del'],
        'llm_thought': result['thought'],
        'llm_evaluation': result['evaluation'],
        'trace': result['trace'],
    }

# --- Output ---
local_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../results/LendingClub")
os.makedirs(local_dir, exist_ok=True)

def get_path(method, model):
    return os.path.join(local_dir, f'{method}_{model.split("/")[-1]}.csv')

METHODS = ["control", "selfcritic", "confidence", "counterfactual", "counterfactual2", "evidence", "control_cot", "control_rf", "control_cot_rf", "prediction_reasoning", "escalate_only", "injected", "reasoning_only", "reasoning_prediction"]
N_SAMPLES = {
    "control": N_SAMPLES_CONTROL,
    "control_cot": N_SAMPLES_CONTROL_COT,
    "control_rf": N_SAMPLES_CONTROL_RF,
    "control_cot_rf": N_SAMPLES_CONTROL_COT_RF,
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
for model, n in [(OAI_MODEL, N_OAI), (OAI_MODEL_NANO, N_NANO), (QWEN_MODEL, N_QWEN)]:
    if n > 0:
        for method in METHODS:
            n_samples = N_SAMPLES[method]
            if n_samples > 0:
                sampled = random.sample(holdout_indices, n * n_samples)
                for idx in sampled:
                    jobs.append((idx, method, model))

print(f"Starting {total} jobs across {len(METHODS)} methods", flush=True)
with ThreadPoolExecutor(max_workers=5) as executor:
    futures = [executor.submit(call_llm_tracked, idx, method, model) for idx, method, model in jobs]
    for f in as_completed(futures):
        f.result()

df_new = pd.DataFrame([r for r in results if r is not None])
for (method, model), group in df_new.groupby(['method', 'model']):
    path = get_path(method, model)
    print(f"Saved to {path}", flush=True)
    print(pd.read_csv(path)[['index', 'llm_prediction', 'human_response', 'llm_escalate', 'method', 'model']].to_string(), flush=True)

# --- History method (sequential) ---
for model, n in [(OAI_MODEL, N_OAI), (OAI_MODEL_NANO, N_NANO), (QWEN_MODEL, N_QWEN)]:
    if n > 0 and N_SAMPLES_HISTORY > 0:
        n_history = n * N_SAMPLES_HISTORY
        sampled = random.sample(holdout_indices, n_history)
        correct = 0
        total_hist = 0
        history_results = []

        path = get_path("history", model)
        try:
            df_hist_existing = pd.read_csv(path)
        except FileNotFoundError:
            df_hist_existing = pd.DataFrame()

        print(f"\n--- Starting history method ({n_history} samples, {model}) ---", flush=True)
        for idx in sampled:
            row = data.loc[idx]
            base = create_prompt_base(row)
            human_response = int(row['accepted'])

            if total_hist > 0:
                pct = correct / total_hist * 100
                prompt = f"So far, you have correctly predicted {correct} out of {total_hist} decisions ({pct:.0f}%).\n\n{base}"
            else:
                prompt = base

            result = get_base_inference(prompt, model)
            pred = result['pred']
            if pred is not None and pred == human_response:
                correct += 1
            total_hist += 1

            row_result = {
                'index': idx,
                'loan_amnt': row['loan_amnt'],
                'purpose': row['purpose'],
                'human_response': human_response,
                'prompt': prompt,
                'method': 'history',
                'model': model,
                'llm_prediction': result['pred'],
                'llm_escalate': result['del'],
                'llm_thought': result['thought'],
                'llm_evaluation': result['evaluation'],
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
