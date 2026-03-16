import os, re, sys, datetime, threading, random
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor, as_completed
import openai

OAI_MODEL      = "gpt-5-mini-2025-08-07"
OAI_MODEL_NANO = "gpt-5-nano-2025-08-07"
QWEN_MODEL     = "Qwen/Qwen3-4B"
if len(sys.argv) > 1:
    QWEN_MODEL = sys.argv[1]

N_SAMPLES_CONTROL           = int(os.environ.get("N_SAMPLES_CONTROL", 50))
N_SAMPLES_CONTROL_COT    = int(os.environ.get("N_SAMPLES_CONTROL_COT", 50))
N_SAMPLES_SELFCRITIC     = int(os.environ.get("N_SAMPLES_SELFCRITIC", 50))
N_SAMPLES_CONFIDENCE     = int(os.environ.get("N_SAMPLES_CONFIDENCE", 50))
N_SAMPLES_COUNTERFACTUAL = int(os.environ.get("N_SAMPLES_COUNTERFACTUAL", 50))
N_SAMPLES_COUNTERFACTUAL2 = int(os.environ.get("N_SAMPLES_COUNTERFACTUAL2", 50))
N_SAMPLES_CONTROL_RF      = int(os.environ.get("N_SAMPLES_CONTROL_RF", 50))
N_SAMPLES_CONTROL_COT_RF  = int(os.environ.get("N_SAMPLES_CONTROL_COT_RF", 50))
N_SAMPLES_EVIDENCE       = int(os.environ.get("N_SAMPLES_EVIDENCE", 50))
N_SAMPLES_PRED_REASONING  = int(os.environ.get("N_SAMPLES_PRED_REASONING", 0))
N_SAMPLES_ESCALATE_ONLY    = int(os.environ.get("N_SAMPLES_ESCALATE_ONLY", 0))
N_SAMPLES_INJECTED         = int(os.environ.get("N_SAMPLES_INJECTED", 0))
N_SAMPLES_REASONING_ONLY       = int(os.environ.get("N_SAMPLES_REASONING_ONLY", 0))
N_SAMPLES_REASONING_PREDICTION = int(os.environ.get("N_SAMPLES_REASONING_PREDICTION", 0))
N_SAMPLES_HISTORY          = int(os.environ.get("N_SAMPLES_HISTORY", 0))
N_OAI  = int(os.environ.get("N_OAI", 0))
N_NANO = int(os.environ.get("N_NANO", 0))
N_QWEN = int(os.environ.get("N_QWEN", 1))

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/MovieLens/movies_and_ratings_last1000000.csv")

# --- Load data ---
print("Loading MovieLens data...", flush=True)
df = pd.read_csv(DATA_PATH)

# --- Train OLS ---
print("Training OLS...", flush=True)
df_ols = df.copy()
df_ols['year'] = (
    df_ols['title']
    .str.extract(r'\((\d{4})\)$', expand=False)
    .pipe(pd.to_numeric, errors='coerce')
    .fillna(0)
    .astype(int)
)
genre_counts = df_ols['genres'].fillna('').str.get_dummies(sep='|').astype('int8')
df_ols = df_ols.join(genre_counts)

formula = (
    "rating ~ year + "
    "Action + Adventure + Animation + Children + Comedy + Crime + "
    "Documentary + Drama + Fantasy + Q('Film-Noir') + Horror + IMAX + "
    "Musical + Mystery + Romance + Q('Sci-Fi') + Thriller + War + Western"
)

df_ols['user_movie_key'] = df_ols['userId'].astype(str) + "_" + df_ols['movieId'].astype(str)
train_df, test_df = train_test_split(df_ols, test_size=0.2, random_state=42)
train_df = train_df.copy(); test_df = test_df.copy()
train_df['split'] = 'train'; test_df['split'] = 'test'

ols = smf.ols(formula=formula, data=train_df).fit()
r2 = ols.rsquared
df_ols['pred'] = ols.predict(df_ols)
print(f"OLS R²: {r2:.3f}", flush=True)

pred_df  = df_ols[['user_movie_key', 'pred']]
split_df = pd.concat([train_df[['user_movie_key', 'split']], test_df[['user_movie_key', 'split']]])

df['user_movie_key'] = df['userId'].astype(str) + "_" + df['movieId'].astype(str)
df = df.merge(pred_df,  on='user_movie_key', how='left')
df = df.merge(split_df, on='user_movie_key', how='left')

test_indices = df.loc[df['split'] == 'test'].index.tolist()

# --- RF baseline ---
rf_features = ['year'] + [c for c in genre_counts.columns if c != '(no genres listed)']
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(df_ols.loc[train_df.index, rf_features], train_df['rating'])
df_ols['rf_pred'] = rf_model.predict(df_ols[rf_features])
rf_r2 = rf_model.score(df_ols.loc[test_df.index, rf_features], test_df['rating'])
print(f"RF baseline R²: {rf_r2:.3f}", flush=True)

rf_pred_df = df_ols[['user_movie_key', 'rf_pred']]
df = df.merge(rf_pred_df, on='user_movie_key', how='left')

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

def create_prompt_base(row_idx):
    user_id = df.iloc[row_idx]['userId']
    user_data = df[df['userId'] == user_id].copy()

    if len(user_data) < 7:
        return None, None, None

    shuffled = user_data.sample(frac=1)
    movie_1 = shuffled.iloc[0]
    movie_2 = None
    used_indices = [0]

    for i in range(1, len(shuffled)):
        if shuffled.iloc[i]['rating'] != movie_1['rating']:
            movie_2 = shuffled.iloc[i]
            used_indices.append(i)
            break

    if movie_2 is None:
        return None, None, None

    history = shuffled.drop(shuffled.index[used_indices]).head(5)

    prompt = "Person 1 has reviewed the following movies:\n\n"
    for _, r in history.iterrows():
        prompt += f"- {r['title']} ({r['genres']}): Rated {r['rating']}/5\n"
    prompt += "\nConsider these two movies they have not seen:\n\n"

    test_pair = [movie_1, movie_2]
    random.shuffle(test_pair)
    for movie in test_pair:
        prompt += f"- {movie['title']} ({movie['genres']})\n"

    answer_key = {movie_1['title']: movie_1['rating'], movie_2['title']: movie_2['rating']}
    pred_key   = {movie_1['title']: movie_1['pred'],   movie_2['title']: movie_2['pred']}
    return prompt, answer_key, pred_key

def create_prompt_rf(base, rf_pred_key):
    """Append RF prediction to the base prompt for MovieLens pairwise comparison."""
    titles = list(rf_pred_key.keys())
    title_1, title_2 = titles[0], titles[1]
    rf_1, rf_2 = rf_pred_key[title_1], rf_pred_key[title_2]
    higher = "first" if rf_1 >= rf_2 else "second"
    base += (
        f" A statistical model predicts Person 1 would rate the first movie {rf_1:.2f}/5 "
        f"and the second movie {rf_2:.2f}/5, suggesting they would prefer the {higher} movie."
    )
    return base

def parse_prediction(text):
    preds = re.findall(r'PREDICTION:\s*([12])', text)
    if preds:
        return int(preds[-1])
    digits = re.findall(r'[12]', text.strip())
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
    "TASK: Predict which movie Person 1 would rate higher. Explain your reasoning in 1 sentence. "
    "Then conclude with exactly 'PREDICTION: 1' (first movie) or 'PREDICTION: 2' (second movie)."
)

# ── Step 1 for confidence: Predict + Reason + Confidence ──
THOUGHT_CONFIDENCE_PROMPT = (
    "TASK: Predict which movie Person 1 would rate higher. Explain your reasoning in 1 sentence. "
    "Rate your confidence from 1 (very uncertain) to 10 (very certain). "
    "Then conclude with exactly 'PREDICTION: 1' (first movie) or 'PREDICTION: 2' (second movie)."
)

# ── Base: single-call prompt (predict + reason + decide) ──
BASE_PROMPT = (
    "TASK: Predict which movie Person 1 would rate higher. Explain your reasoning in 1 sentence. "
    "Then conclude with exactly 'PREDICTION: 1' (first movie) or 'PREDICTION: 2' (second movie).\n"
    "Then, respond with a digit ONLY on a new line.\n"
    "Output 0 to implement your prediction, or 1 to escalate to Person 1.\n"
    "A decision is correct if it matches the movie Person 1 would rate higher."
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
    "A decision is correct if it matches the movie Person 1 would rate higher."
)

ESCALATE_ONLY_TASK = (
    "An LLM was asked to predict which movie Person 1 would rate higher.\n"
    "TASK: Respond with a digit ONLY.\n"
    "Output 0 to implement the LLM's prediction, or 1 to escalate to Person 1.\n"
    "A decision is correct if it matches the movie Person 1 would rate higher."
)

INJECTED_DECISION_TASK = (
    "A separate model predicted PREDICTION: {pred} for which movie Person 1 would rate higher.\n"
    "TASK: Respond with a digit ONLY.\n"
    "Output 0 to implement the model's prediction, or 1 to escalate to Person 1.\n"
    "A decision is correct if it matches the movie Person 1 would rate higher."
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

def call_llm(row_idx, method, model):
    base, answer_key, pred_key = create_prompt_base(row_idx)
    if base is None:
        return None

    titles = list(answer_key.keys())
    title_1, title_2 = titles[0], titles[1]
    rating_1, rating_2 = answer_key[title_1], answer_key[title_2]
    human_response = 1 if rating_1 >= rating_2 else 2
    ols_pred_1, ols_pred_2 = pred_key[title_1], pred_key[title_2]
    user_id    = df.iloc[row_idx]['userId']
    movie_id_1 = df.loc[df['title'] == title_1, 'movieId'].iloc[0]
    movie_id_2 = df.loc[df['title'] == title_2, 'movieId'].iloc[0]

    common = {
        'userId': user_id, 'movieId1': movie_id_1, 'movieId2': movie_id_2,
        'rating_1': rating_1, 'rating_2': rating_2,
        'ols_pred_1': ols_pred_1, 'ols_pred_2': ols_pred_2,
        'human_response': human_response, 'prompt': base,
        'method': method, 'model': model,
    }

    if method in ("control", "control_rf"):
        if method == "control_rf":
            rf_pred_1 = df.loc[df['title'] == title_1, 'rf_pred'].iloc[0]
            rf_pred_2 = df.loc[df['title'] == title_2, 'rf_pred'].iloc[0]
            rf_pred_key = {title_1: rf_pred_1, title_2: rf_pred_2}
            prompt = create_prompt_rf(base, rf_pred_key)
        else:
            prompt = base
        result = get_base_inference(prompt, model)
    elif method in ("control_cot", "control_cot_rf"):
        if method == "control_cot_rf":
            rf_pred_1 = df.loc[df['title'] == title_1, 'rf_pred'].iloc[0]
            rf_pred_2 = df.loc[df['title'] == title_2, 'rf_pred'].iloc[0]
            rf_pred_key = {title_1: rf_pred_1, title_2: rf_pred_2}
            prompt = create_prompt_rf(base, rf_pred_key)
        else:
            prompt = base
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
local_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../results/MovieLens")
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
                sampled = random.sample(test_indices, n * n_samples)
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
    print(pd.read_csv(path)[['userId', 'llm_prediction', 'human_response', 'llm_escalate', 'method', 'model']].to_string(), flush=True)

# --- History method (sequential) ---
for model, n in [(OAI_MODEL, N_OAI), (OAI_MODEL_NANO, N_NANO), (QWEN_MODEL, N_QWEN)]:
    if n > 0 and N_SAMPLES_HISTORY > 0:
        n_history = n * N_SAMPLES_HISTORY
        sampled = random.sample(test_indices, n_history)
        correct = 0
        total_hist = 0
        history_results = []

        path = get_path("history", model)
        try:
            df_hist_existing = pd.read_csv(path)
        except FileNotFoundError:
            df_hist_existing = pd.DataFrame()

        print(f"\n--- Starting history method ({n_history} samples, {model}) ---", flush=True)
        for row_idx in sampled:
            base, answer_key, pred_key = create_prompt_base(row_idx)
            if base is None:
                continue

            titles = list(answer_key.keys())
            title_1, title_2 = titles[0], titles[1]
            rating_1, rating_2 = answer_key[title_1], answer_key[title_2]
            human_response = 1 if rating_1 >= rating_2 else 2

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

            ols_pred_1, ols_pred_2 = pred_key[title_1], pred_key[title_2]
            user_id = df.iloc[row_idx]['userId']
            movie_id_1 = df.loc[df['title'] == title_1, 'movieId'].iloc[0]
            movie_id_2 = df.loc[df['title'] == title_2, 'movieId'].iloc[0]

            row_result = {
                'userId': user_id, 'movieId1': movie_id_1, 'movieId2': movie_id_2,
                'rating_1': rating_1, 'rating_2': rating_2,
                'ols_pred_1': ols_pred_1, 'ols_pred_2': ols_pred_2,
                'human_response': human_response, 'prompt': prompt,
                'method': 'history', 'model': model,
                'llm_prediction': result['pred'],
                'llm_escalate': result['del'],
                'llm_thought': result['thought'],
                'llm_evaluation': result['evaluation'],
                'trace': result['trace'],
                'running_correct': correct,
                'running_total': total_hist,
            }
            history_results.append(row_result)
            print(f"[history {total_hist}/{n_history}] row {row_idx}, pred={pred}, truth={human_response}, acc={correct}/{total_hist}", flush=True)

        if history_results:
            df_hist = pd.DataFrame(history_results)
            df_hist['timestamp'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            pd.concat([df_hist_existing, df_hist], ignore_index=True).to_csv(path, index=False)
            print(f"Saved history to {path}", flush=True)
