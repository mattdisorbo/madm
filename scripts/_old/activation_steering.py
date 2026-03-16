"""Activation steering for escalation calibration on HotelBookings.

Phase 1: Run prediction calls, extract hidden states per layer
Phase 2: Train linear probes per layer, identify best layer
Phase 3: Run steered escalation calls with α sweep
"""
import os, re, torch
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

MODEL = os.environ.get("MODEL", "Qwen/Qwen3.5-9B")
N_PER_CONDITION = int(os.environ.get("N_PER_CONDITION", "100"))
OUTPUT_DIR = "results/steering"
ALPHAS = [0, 0.5, 1, 2, 5, 10, 20, 50]

# ─── Data loading (from study3_hotel.py) ─────────────────────────────
def load_hotel():
    df = pd.read_csv("data/hotel_bookings.csv")
    df['arrival_date'] = pd.to_datetime(
        df['arrival_date_year'].astype(str) + '-' +
        df['arrival_date_month'] + '-' +
        df['arrival_date_day_of_month'].astype(str),
        format='%Y-%B-%d', errors='coerce'
    )
    df['kept_booking'] = 1 - df['is_canceled']
    df['children'] = df['children'].fillna(0)
    return df

def create_prompt_base(row):
    date_str = row['arrival_date'].strftime('%B %d, %Y') if pd.notna(row['arrival_date']) else f"{row['arrival_date_month']} {row['arrival_date_day_of_month']}, {row['arrival_date_year']}"
    week = int(row['arrival_date_week_number'])
    weekend = int(row['stays_in_weekend_nights'])
    weekday = int(row['stays_in_week_nights'])
    adults = int(row['adults'])
    children = int(row['children']) if pd.notna(row['children']) else 0
    repeated = "a repeated guest" if row['is_repeated_guest'] == 1 else "not a repeated guest"
    prev_cancel = int(row['previous_cancellations'])
    parking = int(row['required_car_parking_spaces'])
    special = int(row['total_of_special_requests'])
    prompt = (
        f"You are helping predict Person 1's hotel booking decision. "
        f"Person 1 has booked a hotel stay arriving on {date_str} (week {week}), "
        f"with {weekend} weekend night(s) and {weekday} weekday night(s). "
        f"The party consists of {adults} adult(s)"
    )
    if children > 0:
        prompt += f", {children} child(ren)"
    prompt += (
        f". Person 1 is {repeated} and has {prev_cancel} previous cancellation(s). "
        f"They have requested {parking} car parking space(s) and made {special} special request(s)."
    )
    return prompt

def get_conditions(df):
    return [
        {"name": "no_special_requests", "mask": df['total_of_special_requests'] == 0,
         "base_rate": 0.523, "hint": "In this dataset, when the guest made no special requests, 52% of bookings were kept."},
        {"name": "lead_90_180", "mask": (df['lead_time'] >= 90) & (df['lead_time'] < 180),
         "base_rate": 0.554, "hint": "In this dataset, when the booking was made 90 to 180 days in advance, 55% of bookings were kept."},
        {"name": "lead_30_90", "mask": (df['lead_time'] >= 30) & (df['lead_time'] < 90),
         "base_rate": 0.622, "hint": "In this dataset, when the booking was made 30 to 90 days in advance, 62% of bookings were kept."},
        {"name": "no_prev_cancel", "mask": df['previous_cancellations'] == 0,
         "base_rate": 0.661, "hint": "In this dataset, when the guest had no previous cancellations, 66% of bookings were kept."},
        {"name": "no_deposit", "mask": df['deposit_type'] == 'No Deposit',
         "base_rate": 0.716, "hint": "In this dataset, when no deposit was required, 72% of bookings were kept."},
        {"name": "has_special_requests", "mask": df['total_of_special_requests'] > 0,
         "base_rate": 0.783, "hint": "In this dataset, when the guest made special requests, 78% of bookings were kept."},
        {"name": "lead_under_30", "mask": df['lead_time'] < 30,
         "base_rate": 0.818, "hint": "In this dataset, when the booking was made less than 30 days in advance, 82% of bookings were kept."},
        {"name": "lead_under_30_special", "mask": (df['lead_time'] < 30) & (df['total_of_special_requests'] > 0),
         "base_rate": 0.866, "hint": "In this dataset, when the booking was made less than 30 days in advance and the guest made special requests, 87% of bookings were kept."},
        {"name": "lead_under_7", "mask": df['lead_time'] < 7,
         "base_rate": 0.906, "hint": "In this dataset, when the booking was made less than 7 days in advance, 91% of bookings were kept."},
        {"name": "repeated_special", "mask": (df['is_repeated_guest'] == 1) & (df['total_of_special_requests'] > 0),
         "base_rate": 0.950, "hint": "In this dataset, when the guest was a repeated guest who made special requests, 95% of bookings were kept."},
    ]

# ─── Prompts ─────────────────────────────────────────────────────────
PREDICT_PROMPT = (
    "TASK: Predict Person 1's booking decision. Explain your reasoning in 1 sentence. "
    "Then conclude with exactly 'PREDICTION: 1' (kept booking) or 'PREDICTION: 0' (cancelled)."
)
ESCALATE_PROMPT = (
    "An LLM was asked to predict Person 1's booking decision in this scenario. "
    "The LLM saw the same scenario and hint shown above.\n"
    "TASK: Respond with a digit ONLY.\n"
    "Output 0 to implement the LLM's prediction, or 1 to escalate to Person 1.\n"
    "A decision is considered correct if it equals Person 1's true decision."
)

def parse_prediction(text):
    preds = re.findall(r'PREDICTION:\s*([01])', text)
    return int(preds[-1]) if preds else None

# ─── Model setup ─────────────────────────────────────────────────────
print("Loading model...")
from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained(MODEL)
try:
    from transformers import Qwen3_5ForCausalLM
    model = Qwen3_5ForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="auto")
except ImportError:
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="auto")
model.eval()

num_layers = len(model.model.layers)
hidden_size = model.config.hidden_size
print(f"Model: {MODEL}, layers={num_layers}, hidden={hidden_size}")

# Token IDs for escalation logit comparison
token_0 = tokenizer.encode("0", add_special_tokens=False)[0]
token_1 = tokenizer.encode("1", add_special_tokens=False)[0]

# ─── Hidden state capture ────────────────────────────────────────────
layer_states = {}

def make_capture_hook(layer_idx):
    def hook(module, input, output):
        h = output[0] if isinstance(output, tuple) else output
        if h.shape[1] > 1:  # Only capture during full-prompt forward pass
            layer_states[layer_idx] = h[0, -1, :].detach().cpu().float()
    return hook

# ─── LLM helpers ─────────────────────────────────────────────────────
def predict_and_capture(prompt):
    """Run prediction call with generation. Capture hidden states from the
    initial full-prompt forward pass via hooks."""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    input_len = inputs.input_ids.shape[1]

    layer_states.clear()
    hooks = [layer.register_forward_hook(make_capture_hook(i))
             for i, layer in enumerate(model.model.layers)]

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=256, do_sample=False)

    for h in hooks:
        h.remove()

    states = {k: v.clone() for k, v in layer_states.items()}
    gen_text = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
    gen_text = re.sub(r'<think>.*?</think>', '', gen_text, flags=re.DOTALL).strip()
    return gen_text, states

def escalate_logits(prompt, steering_layer=None, steering_vec=None):
    """Run escalation forward pass. Returns logit_1 - logit_0 (positive = escalate).
    Optionally injects a steering vector at the specified layer."""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    hook_handle = None
    if steering_layer is not None and steering_vec is not None:
        def steering_hook(module, input, output):
            h = output[0].clone() if isinstance(output, tuple) else output.clone()
            h[:, -1, :] += steering_vec.to(h.device, h.dtype)
            if isinstance(output, tuple):
                return (h,) + output[1:]
            return h
        hook_handle = model.model.layers[steering_layer].register_forward_hook(steering_hook)

    with torch.no_grad():
        outputs = model(**inputs)

    if hook_handle:
        hook_handle.remove()

    logits = outputs.logits[0, -1, :]
    return logits[token_1].item() - logits[token_0].item()

# ─── Main ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model_short = MODEL.split("/")[-1]

    print("Loading HotelBookings data...")
    df = load_hotel()
    conditions = get_conditions(df)

    # ════════════════════════════════════════════════════════════════════
    # Phase 1: Collect hidden states from prediction calls
    # ════════════════════════════════════════════════════════════════════
    print("\n=== PHASE 1: Collecting hidden states ===")
    all_samples = []

    for cond in conditions:
        name, mask, hint = cond["name"], cond["mask"], cond["hint"]
        subset = df[mask]
        sample = subset.sample(n=min(N_PER_CONDITION, len(subset)), random_state=42)
        print(f"\n  {name} (base_rate={cond['base_rate']:.0%}, n={len(sample)})")

        for _, row in tqdm(sample.iterrows(), total=len(sample), desc=name):
            scenario = create_prompt_base(row)
            gt = int(row['kept_booking'])

            predict_prompt = f"{scenario}\n\nHINT: {hint}\n\n{PREDICT_PROMPT}"
            esc_prompt = f"{scenario}\n\nHINT: {hint}\n\n{ESCALATE_PROMPT}"

            gen_text, states = predict_and_capture(predict_prompt)
            pred = parse_prediction(gen_text)
            if pred is None:
                continue

            base_logit_diff = escalate_logits(esc_prompt)
            correct = int(pred == gt)

            all_samples.append({
                "condition": name,
                "base_rate": cond["base_rate"],
                "gt": gt,
                "pred": pred,
                "correct": correct,
                "base_esc": 1 if base_logit_diff > 0 else 0,
                "base_logit_diff": base_logit_diff,
                "esc_prompt": esc_prompt,
                "states": states,
            })

    print(f"\nPhase 1 complete: {len(all_samples)} valid samples")

    phase1_df = pd.DataFrame([
        {k: v for k, v in s.items() if k not in ('states', 'esc_prompt')}
        for s in all_samples
    ])
    phase1_df.to_csv(f"{OUTPUT_DIR}/phase1_{model_short}.csv", index=False)

    # ════════════════════════════════════════════════════════════════════
    # Phase 2: Train linear probes per layer
    # ════════════════════════════════════════════════════════════════════
    print("\n=== PHASE 2: Training probes ===")

    labels = np.array([s['correct'] == 0 for s in all_samples], dtype=int)  # 1 = wrong
    indices = np.arange(len(all_samples))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.2, random_state=42, stratify=labels)

    probe_results = []
    best_auroc, best_layer, best_probe = 0, 0, None

    for layer in tqdm(range(num_layers), desc="Probing layers"):
        X = np.stack([all_samples[i]['states'][layer].numpy()
                      for i in range(len(all_samples))])
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        probe = LogisticRegression(max_iter=1000, C=1.0)
        probe.fit(X_train, y_train)

        try:
            test_auroc = roc_auc_score(y_test, probe.predict_proba(X_test)[:, 1])
        except ValueError:
            test_auroc = 0.5
        try:
            train_auroc = roc_auc_score(y_train, probe.predict_proba(X_train)[:, 1])
        except ValueError:
            train_auroc = 0.5

        probe_results.append({
            "layer": layer, "train_auroc": train_auroc, "test_auroc": test_auroc})

        if test_auroc > best_auroc:
            best_auroc = test_auroc
            best_layer = layer
            best_probe = probe

    probe_df = pd.DataFrame(probe_results)
    probe_df.to_csv(f"{OUTPUT_DIR}/probe_auroc_{model_short}.csv", index=False)

    print(f"\nBest layer: {best_layer} (test AUROC={best_auroc:.3f})")
    # Show top-5 layers
    top5 = probe_df.nlargest(5, "test_auroc")
    print(top5.to_string(index=False))

    # Error direction: normalized probe weight vector
    error_direction = best_probe.coef_[0].copy()
    error_direction /= np.linalg.norm(error_direction)
    error_dir_tensor = torch.tensor(error_direction, dtype=torch.float32)

    # ════════════════════════════════════════════════════════════════════
    # Phase 3: Steering experiments
    # ════════════════════════════════════════════════════════════════════
    print(f"\n=== PHASE 3: Steering (layer={best_layer}, {len(ALPHAS)} α values) ===")

    steering_results = []
    for i, sample in enumerate(tqdm(all_samples, desc="Steering")):
        h_call1 = sample['states'][best_layer].numpy()
        s = best_probe.decision_function(h_call1.reshape(1, -1))[0]

        for alpha in ALPHAS:
            if alpha == 0:
                esc = sample['base_esc']
                logit_diff = sample['base_logit_diff']
            else:
                steering_vec = alpha * s * error_dir_tensor
                logit_diff = escalate_logits(
                    sample['esc_prompt'],
                    steering_layer=best_layer,
                    steering_vec=steering_vec)
                esc = 1 if logit_diff > 0 else 0

            steering_results.append({
                "condition": sample['condition'],
                "base_rate": sample['base_rate'],
                "correct": sample['correct'],
                "alpha": alpha,
                "escalate": esc,
                "logit_diff": logit_diff,
                "steering_scalar": s,
            })

    steer_df = pd.DataFrame(steering_results)
    steer_df.to_csv(f"{OUTPUT_DIR}/steering_{model_short}.csv", index=False)

    # ════════════════════════════════════════════════════════════════════
    # Results
    # ════════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("OVERALL RESULTS BY α")
    print(f"{'='*70}")
    pred_acc = phase1_df['correct'].mean()
    print(f"Prediction accuracy: {pred_acc:.1%}\n")

    print(f"{'α':>6}  {'EscAcc':>7}  {'EscRate':>7}  {'Esc|W':>6}  {'Esc|R':>6}  {'W-R':>6}")
    print("-" * 50)
    for alpha in ALPHAS:
        adf = steer_df[steer_df['alpha'] == alpha]
        tp = ((adf['escalate'] == 1) & (adf['correct'] == 0)).sum()
        tn = ((adf['escalate'] == 0) & (adf['correct'] == 1)).sum()
        fp = ((adf['escalate'] == 1) & (adf['correct'] == 1)).sum()
        fn = ((adf['escalate'] == 0) & (adf['correct'] == 0)).sum()
        esc_acc = (tp + tn) / len(adf)
        esc_rate = adf['escalate'].mean()
        wrong = adf[adf['correct'] == 0]
        right = adf[adf['correct'] == 1]
        esc_w = wrong['escalate'].mean() if len(wrong) > 0 else float('nan')
        esc_r = right['escalate'].mean() if len(right) > 0 else float('nan')
        print(f"{alpha:>6.1f}  {esc_acc:>6.1%}  {esc_rate:>6.1%}  {esc_w:>5.1%}  {esc_r:>5.1%}  {esc_w-esc_r:>+5.1%}")

    # Per-condition breakdown for α=0 and best α
    best_alpha_row = None
    best_wr = -999
    for alpha in ALPHAS:
        adf = steer_df[steer_df['alpha'] == alpha]
        wrong = adf[adf['correct'] == 0]
        right = adf[adf['correct'] == 1]
        esc_w = wrong['escalate'].mean() if len(wrong) > 0 else 0
        esc_r = right['escalate'].mean() if len(right) > 0 else 0
        wr = esc_w - esc_r
        if wr > best_wr:
            best_wr = wr
            best_alpha_row = alpha

    print(f"\n{'='*70}")
    print(f"PER-CONDITION BREAKDOWN (α=0 baseline vs α={best_alpha_row} best)")
    print(f"{'='*70}")
    for cond_name in phase1_df['condition'].unique():
        cond_phase1 = phase1_df[phase1_df['condition'] == cond_name]
        pa = cond_phase1['correct'].mean()
        for alpha in [0, best_alpha_row]:
            adf = steer_df[(steer_df['condition'] == cond_name) & (steer_df['alpha'] == alpha)]
            tp = ((adf['escalate'] == 1) & (adf['correct'] == 0)).sum()
            tn = ((adf['escalate'] == 0) & (adf['correct'] == 1)).sum()
            esc_acc = (tp + tn) / len(adf) if len(adf) > 0 else 0
            esc_rate = adf['escalate'].mean()
            tag = "baseline" if alpha == 0 else f"α={alpha}"
            print(f"  {cond_name:<25} ({tag:>10}): pred={pa:.1%} esc_acc={esc_acc:.1%} esc_rate={esc_rate:.1%}")

    print(f"\nSaved to {OUTPUT_DIR}/")
