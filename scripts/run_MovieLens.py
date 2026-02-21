import os, re, datetime, threading, random
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor, as_completed
import openai

OAI_MODEL  = "gpt-5-mini-2025-08-07"
QWEN_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

N_SAMPLES_BASE    = 1
N_SAMPLES_AUDITOR = 0
N_OAI  = 1
N_QWEN = 0

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

# --- Qwen setup ---
qwen_pipe = None
qwen_lock = threading.Lock()

if N_QWEN > 0:
    from transformers import pipeline
    print(f"Loading {QWEN_MODEL}...", flush=True)
    qwen_pipe = pipeline("text-generation", model=QWEN_MODEL, torch_dtype="auto", device_map="auto")
    print("Qwen loaded.", flush=True)

def llm(prompt, model):
    if model == QWEN_MODEL:
        with qwen_lock:
            out = qwen_pipe([{"role": "user", "content": prompt}], max_new_tokens=2048)
        return out[0]["generated_text"][-1]["content"]
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

def get_llm_base(base_prompt, model):
    instructions = (
        "\n\nRespond with two numbers ONLY on two lines. "
        "First, respond 1 (first movie) or 2 (second movie) based on your prediction of which movie Person 1 would rate higher.\n"
        "Second, respond 0 to recommend the movie you predicted, "
        "or 1 to delegate to Person 1 so they can choose. Person 1 will always choose the movie they would rate higher.\n"
        "A decision is correct if it is the movie Person 1 would rate higher."
    )
    full_prompt = base_prompt + instructions
    response = llm(full_prompt, model)
    lines = [l.strip() for l in response.strip().split('\n') if l.strip()]
    try:
        pred = int(re.search(r'[12]', lines[0]).group()) if lines else None
        delg = int(re.search(r'[01]', lines[-1]).group()) if len(lines) > 1 else None
        return {"pred": pred, "del": delg, "full_prompt": full_prompt, "response": response}
    except (ValueError, IndexError, AttributeError):
        print(f"Parse error: {response}", flush=True)
        return {"pred": None, "del": None, "full_prompt": full_prompt, "response": response}

def get_sequential_inference(base_prompt, model):
    try:
        thought = llm(
            f"{base_prompt}\n\n"
            "TASK: Predict which movie Person 1 would rate higher.\n"
            "Explain your reasoning in 1 sentence. Then conclude with exactly:\n"
            "'PREDICTION: 1' (first movie) or 'PREDICTION: 2' (second movie).",
            model
        )
        preds = re.findall(r'PREDICTION:\s*([12])', thought)
        pred = int(preds[-1]) if preds else 1

        critique = llm(
            f"PROMPT:\n{base_prompt}\n\nPROPOSED LOGIC:\n{thought}\n\n"
            "TASK: Critique this reasoning in 1 sentence.",
            model
        )

        decision = llm(
            f"PROMPT:\n{base_prompt}\n\nPREDICTION & REASONING:\n{thought}\n\n"
            f"CRITIQUE:\n{critique}\n\n"
            "TASK: Respond with two numbers ONLY on two lines.\n"
            "Line 1: output 1 or 2 for which movie Person 1 would rate higher.\n"
            "Line 2: output 0 to implement your prediction, or 1 to delegate to Person 1.\n"
            "A decision is correct if it matches the movie Person 1 would rate higher.",
            model
        )
        lines = [l.strip() for l in decision.split('\n') if l.strip()]
        final_pred = int(re.findall(r'[12]', lines[0])[0]) if lines and re.findall(r'[12]', lines[0]) else pred
        final_del  = int(re.findall(r'[01]', lines[1])[0]) if len(lines) > 1 and re.findall(r'[01]', lines[1]) else 1
        return {"full_thought": thought, "pred": final_pred, "critique": critique, "decision": decision, "del": final_del}
    except Exception as e:
        return {"full_thought": str(e), "pred": None, "critique": None, "decision": None, "del": None}

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

    if method == "base":
        result = get_llm_base(base, model)
        trace = f"[PROMPT]\n{result['full_prompt']}\n\n[RESPONSE]\n{result['response']}"
        return {**common, 'llm_prediction': result['pred'], 'llm_delegate': result['del'], 'trace': trace}
    elif method == "auditor":
        result = get_sequential_inference(base, model)
        trace = (f"[PROMPT]\n{base}\n\n[THOUGHT]\n{result['full_thought']}\n\n"
                 f"[CRITIQUE]\n{result['critique']}\n\n[DECISION]\n{result['decision']}")
        return {**common, 'llm_prediction': result['pred'], 'llm_delegate': result['del'],
                'llm_full_thought': result['full_thought'], 'llm_critique': result['critique'], 'trace': trace}

# --- Output ---
local_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../results/MovieLens")
os.makedirs(local_dir, exist_ok=True)

def get_path(method, model):
    return os.path.join(local_dir, f'{method}_{model.split("/")[-1]}.csv')

df_existing = {}
for model, n in [(OAI_MODEL, N_OAI), (QWEN_MODEL, N_QWEN)]:
    if n > 0:
        for method in ["base", "auditor"]:
            path = get_path(method, model)
            try:
                df_existing[(method, model)] = pd.read_csv(path)
            except FileNotFoundError:
                df_existing[(method, model)] = pd.DataFrame()

test_indices = df.loc[df['split'] == 'test'].index.tolist()

results = []
completed = 0
total = (N_OAI + N_QWEN) * (N_SAMPLES_BASE + N_SAMPLES_AUDITOR)
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
for model, n in [(OAI_MODEL, N_OAI), (QWEN_MODEL, N_QWEN)]:
    if n > 0:
        for method, n_samples in [("base", N_SAMPLES_BASE), ("auditor", N_SAMPLES_AUDITOR)]:
            if n_samples > 0:
                sampled = random.sample(test_indices, n * n_samples)
                for idx in sampled:
                    jobs.append((idx, method, model))

print(f"Starting {total} jobs | OAI {N_OAI}x(b={N_SAMPLES_BASE}, a={N_SAMPLES_AUDITOR}) | Qwen {N_QWEN}x(b={N_SAMPLES_BASE}, a={N_SAMPLES_AUDITOR})", flush=True)
with ThreadPoolExecutor(max_workers=5) as executor:
    futures = [executor.submit(call_llm_tracked, idx, method, model) for idx, method, model in jobs]
    for f in as_completed(futures):
        f.result()

df_new = pd.DataFrame([r for r in results if r is not None])
for (method, model), group in df_new.groupby(['method', 'model']):
    path = get_path(method, model)
    print(f"Saved to {path}", flush=True)
    print(pd.read_csv(path)[['userId', 'llm_prediction', 'human_response', 'llm_delegate', 'method', 'model']].to_string(), flush=True)
