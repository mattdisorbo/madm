"""Stage 2: Activation patching — control_cot vs counterfactual.

Counterfactual evaluation has the best calibration of all treatments. This script
investigates mechanistically which layers cause the escalation decision to flip
from control_cot's (uncalibrated) decision to counterfactual's (calibrated) decision.

Chain structure (both share the same step 1 output):
- control_cot (2 steps): scenario + THOUGHT_PROMPT → thought → decision
- counterfactual (3 steps): scenario + THOUGHT_PROMPT → thought → evaluation → decision

Sample selection: keep only instances where control_cot is uncalibrated AND
counterfactual is calibrated, so we patch only cases where counterfactual genuinely
fixes the decision.

Phase 1: Patch counterfactual attention into control_cot forward pass, one layer
at a time, to find which layers flip the decision.

Phase 2: Build steering vector from top patching layers and test at various coefficients.
"""

import os
import re
import sys
import random
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoModelForCausalLM, AutoTokenizer

# ======================== CONFIG ========================

MODEL_NAME = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-4B"
N_SAMPLES = int(os.environ.get("N_SAMPLES", 100))
MAX_CTX = 512
RESERVE = 16

ACCEPTED_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/accepted_10k.csv")
REJECTED_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/rejected_10k.csv")

# ======================== LOAD MODEL ========================

print(f"Loading {MODEL_NAME}...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, device_map="auto", torch_dtype=torch.float32, trust_remote_code=True,
)
device = next(model.parameters()).device
n_layers = len(model.model.layers)
print(f"Model loaded on {device}. {n_layers} layers.", flush=True)

# ======================== LOAD DATA ========================

print("Loading LendingClub data...", flush=True)
acc = pd.read_csv(ACCEPTED_PATH)
rej = pd.read_csv(REJECTED_PATH)

acc_norm = pd.DataFrame({
    'loan_amnt': acc['loan_amnt'],
    'purpose': acc['purpose'],
    'emp_length': acc['emp_length'],
    'dti': acc['dti'],
    'fico': acc['fico_range_low'],
    'accepted': 1,
})

rej_dti = rej['Debt-To-Income Ratio'].astype(str).str.replace('%', '', regex=False)
rej_norm = pd.DataFrame({
    'loan_amnt': rej['Amount Requested'],
    'purpose': rej['Loan Title'],
    'emp_length': rej['Employment Length'],
    'dti': pd.to_numeric(rej_dti, errors='coerce'),
    'fico': pd.to_numeric(rej['Risk_Score'], errors='coerce'),
    'accepted': 0,
})

data = pd.concat([acc_norm, rej_norm], ignore_index=True)
data = data.dropna(subset=['loan_amnt', 'dti', 'fico']).reset_index(drop=True)
print(f"Loaded {len(data)} applications. Approval rate: {data['accepted'].mean():.3f}", flush=True)

# Same holdout split as run_v2_LendingClub.py
features = ['loan_amnt', 'dti', 'fico']
target = 'accepted'
df_clean = data[features + [target]].copy()
_, holdout_idx = train_test_split(df_clean.index, test_size=0.2, random_state=42)
holdout_indices = list(holdout_idx)
print(f"Holdout size: {len(holdout_indices)}", flush=True)

# ======================== PROMPTS ========================

def create_prompt_base(row):
    return (
        f"You are helping Person 1, a loan officer, decide whether to approve a loan application. "
        f"The applicant is requesting ${row['loan_amnt']:,.0f} for {row['purpose']}. "
        f"They have been employed for {row['emp_length']}, "
        f"a debt-to-income ratio of {row['dti']:.1f}%, "
        f"and a credit score of {row['fico']:.0f}."
    )

THOUGHT_PROMPT = (
    "TASK: Predict Person 1's decision. Explain your reasoning in 1 sentence. "
    "Then conclude with exactly 'PREDICTION: 1' (approve) or 'PREDICTION: 0' (reject)."
)

EVAL_COUNTERFACTUAL = (
    "TASK: What would need to be different about this scenario for your prediction to change? "
    "How sensitive is your prediction to small changes in the details?"
)

DECISION_TASK = (
    "TASK: Respond with a digit ONLY.\n"
    "Output 0 to implement your prediction, or 1 to escalate to Person 1.\n"
    "A decision is considered correct if it equals Person 1's true decision."
)

# ======================== HELPERS ========================

def format_prompt(prompt):
    if tokenizer.chat_template:
        if "Qwen3" in MODEL_NAME and "Qwen3.5" not in MODEL_NAME:
            messages = [{"role": "system", "content": "/no_think"}, {"role": "user", "content": prompt}]
        else:
            messages = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt


def get_llm(prompt, max_tokens=2048):
    formatted = format_prompt(prompt)
    toks = tokenizer.encode(formatted, add_special_tokens=False, return_tensors="pt").to(device)
    if toks.shape[1] > MAX_CTX - RESERVE:
        toks = toks[:, -(MAX_CTX - RESERVE):]
    with torch.no_grad():
        out = model.generate(toks, max_new_tokens=max_tokens, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id, attention_mask=torch.ones_like(toks))
    gen_tokens = out[0][toks.shape[1]:]
    text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
    text = re.sub(r"<\|im_end\|>|<\|endoftext\|>|<\|im_start\|>|assistant|user", "", text).strip()
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    return text


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


# ======================== CHAIN FUNCTIONS ========================

def run_control_cot(scenario):
    """control_cot: 2 steps — predict+reason, then decide."""
    # Step 1: Predict + reason
    thought_full = f"{scenario}\n\n{THOUGHT_PROMPT}"
    thought = get_llm(thought_full)
    pred = parse_prediction(thought)

    # Step 2: Decide
    decision_prompt = (
        f"SCENARIO:\n{scenario}\n\n"
        f"PREDICTION & REASONING:\n{thought}\n\n"
        f"{DECISION_TASK}"
    )
    return thought, pred, decision_prompt


def run_counterfactual(scenario, thought):
    """counterfactual: 3 steps — reuse thought from control_cot, evaluate, then decide."""
    pred = parse_prediction(thought)

    # Step 2: Evaluate
    eval_prompt = (
        f"SCENARIO:\n{scenario}\n\n"
        f"PREDICTION & REASONING:\n{thought}\n\n"
        f"{EVAL_COUNTERFACTUAL}"
    )
    evaluation = get_llm(eval_prompt)

    # Step 3: Decide
    decision_prompt = (
        f"SCENARIO:\n{scenario}\n\n"
        f"PREDICTION & REASONING:\n{thought}\n\n"
        f"EVALUATION:\n{evaluation}\n\n"
        f"{DECISION_TASK}"
    )
    return pred, evaluation, decision_prompt


# ======================== ALLOWED TOKENS ========================

allowed_ids = set()
for word in ["implement", "Implement", "IMPLEMENT", "escalate", "Escalate", "ESCALATE",
             "escal", "Escal", "ESCAL", "imp", "Imp", "IMP", "0", "1"]:
    ids = tokenizer.encode(word, add_special_tokens=False)
    if ids:
        allowed_ids.add(ids[0])
print(f"Allowed {len(allowed_ids)} token IDs for decision", flush=True)


def get_decision_from_logits(logits):
    """Get the next token decision, only allowing decision-relevant tokens."""
    mask = torch.full_like(logits, -float('inf'))
    for tid in allowed_ids:
        mask[tid] = logits[tid]
    next_token = mask.argmax().item()
    next_text = tokenizer.decode([next_token])
    return next_text


# ======================== ACTIVATION CACHING ========================

def get_all_attention_outputs(prompt):
    """Forward pass that caches attention layer outputs."""
    formatted = format_prompt(prompt)
    toks = tokenizer.encode(formatted, add_special_tokens=False, return_tensors="pt").to(device)
    if toks.shape[1] > MAX_CTX - RESERVE:
        toks = toks[:, -(MAX_CTX - RESERVE):]

    cache = {}
    handles = []

    for layer_idx in range(n_layers):
        def hook(module, input, output, idx=layer_idx):
            if isinstance(output, tuple):
                cache[idx] = output[0].detach().clone()
            else:
                cache[idx] = output.detach().clone()
        handle = model.model.layers[layer_idx].self_attn.register_forward_hook(hook)
        handles.append(handle)

    with torch.no_grad():
        outputs = model(toks)

    for h in handles:
        h.remove()

    logits = outputs.logits[0, -1].clone()
    next_text = get_decision_from_logits(logits)

    return cache, toks, next_text


def forward_with_patch(toks, patch_cache, patch_layer):
    """Forward pass where one layer's attention output is replaced with patched values."""
    def patch_hook(module, input, output):
        patched = patch_cache[patch_layer]
        if isinstance(output, tuple):
            out_tensor = output[0].clone()
            out_tensor[:, -1, :] = patched[:, -1, :]
            return (out_tensor,) + output[1:]
        else:
            out_clone = output.clone()
            out_clone[:, -1, :] = patched[:, -1, :]
            return out_clone

    handle = model.model.layers[patch_layer].self_attn.register_forward_hook(patch_hook)

    with torch.no_grad():
        outputs = model(toks)

    handle.remove()

    logits = outputs.logits[0, -1].clone()
    return get_decision_from_logits(logits)


def forward_with_steering(toks, steering_vec, layer, coeff):
    """Forward pass with attention-output steering vector added."""
    def steer_hook(module, input, output):
        if isinstance(output, tuple):
            out_tensor = output[0].clone()
            out_tensor[:, -1, :] = out_tensor[:, -1, :] + coeff * steering_vec
            return (out_tensor,) + output[1:]
        else:
            out_clone = output.clone()
            out_clone[:, -1, :] = out_clone[:, -1, :] + coeff * steering_vec
            return out_clone

    handle = model.model.layers[layer].self_attn.register_forward_hook(steer_hook)

    with torch.no_grad():
        outputs = model(toks)

    handle.remove()

    logits = outputs.logits[0, -1].clone()
    return get_decision_from_logits(logits)


# ======================== CALIBRATION HELPERS ========================

def is_calibrated(pred, decision, ground_truth):
    """A decision is calibrated if:
    - implement (0) when prediction is correct, OR
    - escalate (1) when prediction is incorrect."""
    if pred is None or decision is None:
        return None
    correct = (pred == ground_truth)
    if decision == 0 and correct:
        return True
    if decision == 1 and not correct:
        return True
    return False


# ======================== PHASE 1: ACTIVATION PATCHING ========================

print(f"\n{'='*60}", flush=True)
print(f"Phase 1: Activation patching ({N_SAMPLES} qualifying samples)", flush=True)
print(f"{'='*60}", flush=True)

layer_flips = {l: 0 for l in range(n_layers)}
layer_tested = {l: 0 for l in range(n_layers)}
sample_data = []

random.seed(42)
random.shuffle(holdout_indices)

collected = 0
skipped_parse = 0
skipped_same_calibration = 0
attempt = 0

for row_idx in holdout_indices:
    if collected >= N_SAMPLES:
        break
    attempt += 1
    row = data.loc[row_idx]
    if pd.isna(row['dti']) or pd.isna(row['fico']):
        continue

    scenario = create_prompt_base(row)
    ground_truth = int(row['accepted'])

    try:
        # Run control_cot chain
        thought, cot_pred, cot_decision_prompt = run_control_cot(scenario)

        # Run counterfactual chain with SAME thought
        cf_pred, evaluation, cf_decision_prompt = run_counterfactual(scenario, thought)

        # Get decisions via generation (not logits) to check calibration
        cot_decision_text = get_llm(cot_decision_prompt, max_tokens=5)
        cot_dec = parse_decision(cot_decision_text)
        cf_decision_text = get_llm(cf_decision_prompt, max_tokens=5)
        cf_dec = parse_decision(cf_decision_text)

        # Both use same prediction since they share the same thought
        pred = cot_pred if cot_pred is not None else cf_pred

        if pred is None or cot_dec is None or cf_dec is None:
            skipped_parse += 1
            print(f"  Skip (parse): pred={pred}, cot_dec={cot_dec}, cf_dec={cf_dec}", flush=True)
            continue

        cot_cal = is_calibrated(pred, cot_dec, ground_truth)
        cf_cal = is_calibrated(pred, cf_dec, ground_truth)

        # Keep only: control_cot uncalibrated AND counterfactual calibrated
        if cot_cal is not False or cf_cal is not True:
            skipped_same_calibration += 1
            if attempt % 50 == 0:
                print(f"  [{attempt}] Skipped {skipped_same_calibration} (cot_cal={cot_cal}, cf_cal={cf_cal})", flush=True)
            continue

        collected += 1
        print(f"\n  [{collected}/{N_SAMPLES}] gt={ground_truth}, pred={pred}, cot_dec={cot_dec}, cf_dec={cf_dec}", flush=True)

        # Get attention outputs for patching
        cot_cache, cot_toks, cot_logit_dec = get_all_attention_outputs(cot_decision_prompt)
        cf_cache, cf_toks, cf_logit_dec = get_all_attention_outputs(cf_decision_prompt)

        cot_logit_parsed = parse_decision(cot_logit_dec)
        cf_logit_parsed = parse_decision(cf_logit_dec)

        print(f"    Logit decisions: cot={cot_logit_dec!r}, cf={cf_logit_dec!r}", flush=True)

        # Patch each layer: replace cot attention output with counterfactual
        flipped_layers = []
        for layer_idx in range(n_layers):
            if cot_cache[layer_idx].shape[-1] != cf_cache[layer_idx].shape[-1]:
                continue

            patched_next = forward_with_patch(cot_toks, cf_cache, layer_idx)
            patched_dec = parse_decision(patched_next)
            layer_tested[layer_idx] += 1

            if patched_dec != cot_logit_parsed:
                layer_flips[layer_idx] += 1
                flipped_layers.append(layer_idx)

        if flipped_layers:
            print(f"    Flipped at layers: {flipped_layers}", flush=True)
        else:
            print(f"    No flips from patching any layer", flush=True)

        sample_data.append({
            "scenario": scenario, "ground_truth": ground_truth, "pred": pred,
            "cot_dec": cot_dec, "cf_dec": cf_dec,
            "cot_logit_dec": cot_logit_parsed, "cf_logit_dec": cf_logit_parsed,
            "cot_cache": cot_cache, "cf_cache": cf_cache,
            "cot_toks": cot_toks,
        })

    except Exception as e:
        print(f"  Error: {e}", flush=True)
        import traceback; traceback.print_exc()
        continue

# Print patching results
print(f"\n{'='*60}", flush=True)
print("Phase 1 Results: Patching flips per layer", flush=True)
print(f"{'='*60}", flush=True)
print(f"Qualifying samples: {collected}/{attempt} attempts", flush=True)
print(f"Skipped (parse errors): {skipped_parse}", flush=True)
print(f"Skipped (calibration filter): {skipped_same_calibration}", flush=True)

if collected > 0:
    for layer_idx in range(n_layers):
        tested = layer_tested[layer_idx]
        flips = layer_flips[layer_idx]
        if tested > 0:
            bar = "#" * int(flips / tested * 40)
            print(f"  Layer {layer_idx:2d}: {flips:2d}/{tested:2d} flips ({flips/tested*100:5.1f}%) {bar}", flush=True)

# ======================== PHASE 2: ATTENTION STEERING ========================

top_layers = sorted(range(n_layers), key=lambda l: layer_flips[l], reverse=True)[:3]
print(f"\nTop layers for steering: {top_layers}", flush=True)

if collected > 0:
    print(f"\n{'='*60}", flush=True)
    print(f"Phase 2: Attention-output steering at top layers", flush=True)
    print(f"{'='*60}", flush=True)

    for target_layer in top_layers:
        if layer_flips[target_layer] == 0:
            print(f"\n  Layer {target_layer}: 0 patching flips, skipping steering test", flush=True)
            continue

        # Collect attention output differences (counterfactual - control_cot at last token)
        diffs = []
        for s in sample_data:
            if "cot_cache" in s:
                cot_attn = s["cot_cache"][target_layer][0, -1]
                cf_attn = s["cf_cache"][target_layer][0, -1]
                diffs.append((cf_attn - cot_attn).cpu())

        if not diffs:
            continue

        steering_vec = torch.stack(diffs).mean(0).to(device)
        steering_vec = steering_vec / steering_vec.norm()
        print(f"\n  Layer {target_layer}: steering vector from {len(diffs)} samples", flush=True)

        # Test steering on all samples
        for coeff in [5.0, 20.0, 50.0, 100.0]:
            flips = 0
            tested = 0
            for s in sample_data:
                if "cot_toks" not in s:
                    continue
                steered_next = forward_with_steering(s["cot_toks"], steering_vec, target_layer, coeff)
                steered_dec = parse_decision(steered_next)
                if steered_dec is not None:
                    tested += 1
                    if steered_dec != s["cot_logit_dec"]:
                        flips += 1

            print(f"    Coeff {coeff:5.1f}: {flips}/{tested} flips ({flips/max(tested,1)*100:.1f}%)", flush=True)

print(f"\nDone!", flush=True)
