"""Stage 2: Activation patching at attention outputs to find causal layers.

For each sample:
1. Run same scenario through base chain (support) and adversarial chain (critique)
2. At the escalation step, do forward passes for both, caching attention outputs per layer
3. Patch adversarial attention output into base forward pass, one layer at a time
4. Check which layers flip the decision

Then build a steering vector from the most effective layer's attention output difference.
"""

import os
import re
import csv
from datetime import datetime
from functools import partial
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ======================== CONFIG ========================

MODEL_NAME = "Qwen/Qwen3-1.7B"
N_SAMPLES = 15
MAX_CTX = 512
RESERVE = 16

ACCEPTED_CSV = "data/accepted_10k.csv"
REJECTED_CSV = "data/rejected_10k.csv"
OUTPUT_CSV = "results/stage2_attention_patching.csv"

# ======================== LOAD MODEL ========================

print(f"Loading {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, device_map="auto", torch_dtype=torch.float32, trust_remote_code=True,
)
device = next(model.parameters()).device
n_layers = len(model.model.layers)
print(f"Model loaded on {device}. {n_layers} layers.")

# ======================== LOAD DATA ========================

df_accepted = pd.read_csv(ACCEPTED_CSV)
df_rejected = pd.read_csv(REJECTED_CSV)
df = pd.concat([
    df_accepted[["loan_amnt", "title", "addr_state", "emp_length", "policy_code"]].assign(accepted=1),
    df_rejected.rename(columns={
        "Amount Requested": "loan_amnt", "Loan Title": "title",
        "State": "addr_state", "Employment Length": "emp_length", "Policy Code": "policy_code",
    })[["loan_amnt", "title", "addr_state", "emp_length", "policy_code"]].assign(accepted=0),
], ignore_index=True)
df["title"] = df["title"].str.lower().str.replace("_", " ", regex=False)
df["emp_length"] = df["emp_length"].map({
    "< 1 year": 0, "1 year": 1, "2 years": 2, "3 years": 3,
    "4 years": 4, "5 years": 5, "6 years": 6, "7 years": 7,
    "8 years": 8, "9 years": 9, "10+ years": 10,
})

# ======================== HELPERS ========================

def create_prompt_base(row):
    loan_amnt = int(round(row["loan_amnt"]))
    emp_length = int(round(row["emp_length"]))
    base = f"Someone from {row['addr_state']} is applying for a loan of ${loan_amnt}."
    if pd.notna(row["title"]):
        base += f" The loan is for a {row['title']}."
    base += f" They have been employed for {emp_length} years."
    return base


def format_prompt(prompt):
    if tokenizer.chat_template:
        messages = [{"role": "system", "content": "/no_think"}, {"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt


def get_llm_base(prompt, max_tokens=20):
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


def parse_decision(text):
    text_lower = text.lower()
    m = re.findall(r"\b(implement(?:ed|ing|ation)?|escal(?:at(?:e|ed|ing|ion))?)\b", text_lower)
    if m:
        return "implement" if m[-1].startswith("implement") else "escalate"
    return "unknown"


def get_base_chain(scenario):
    initial_prompt = (f"{scenario}\n\nWould you accept or reject this loan application?\n"
                      "You must respond with ONLY one word: either 'accept' or 'reject'. No explanation.\nAnswer:")
    initial_text = get_llm_base(initial_prompt, max_tokens=5)
    support_prompt = (f"{scenario}\n\nInitial decision: {initial_text}\n\n"
                      f"Please list reasons to support this decision to {initial_text.lower()} the loan application.\n1.")
    support_text = get_llm_base(support_prompt, max_tokens=150)
    final_prompt = (f"{scenario}\n\nInitial decision: {initial_text}\n"
                    f"Supporting reasons: {support_text}\n\n"
                    "Based on the reasons above, are you confident enough to implement this decision yourself, "
                    "or should it be escalated to a human for final review?\n"
                    "You must respond with ONLY one word: either 'implement' or 'escalate'. No explanation.")
    return initial_text, support_text, final_prompt


def get_adversarial_chain(scenario):
    initial_prompt = (f"{scenario}\n\nWould you accept or reject this loan application?\n"
                      "You must respond with ONLY one word: either 'accept' or 'reject'. No explanation.\nAnswer:")
    initial_text = get_llm_base(initial_prompt, max_tokens=5)
    critique_prompt = (f"{scenario}\n\nInitial decision: {initial_text}\n\n"
                       "Please list potential concerns or reasons to reconsider.\n1.")
    critique_text = get_llm_base(critique_prompt, max_tokens=150)
    final_prompt = (f"{scenario}\n\nInitial decision: {initial_text}\n"
                    f"Concerns: {critique_text}\n\n"
                    "Based on the reasons above, are you confident enough to implement this decision yourself, "
                    "or should it be escalated to a human for final review?\n"
                    "You must respond with ONLY one word: either 'implement' or 'escalate'. No explanation.")
    return initial_text, critique_text, final_prompt


# ======================== ACTIVATION CACHING ========================

def get_all_attention_outputs(prompt):
    """Forward pass that caches attention layer outputs (the output of the full attention block)."""
    formatted = format_prompt(prompt)
    toks = tokenizer.encode(formatted, add_special_tokens=False, return_tensors="pt").to(device)
    if toks.shape[1] > MAX_CTX - RESERVE:
        toks = toks[:, -(MAX_CTX - RESERVE):]

    cache = {}
    handles = []

    for layer_idx in range(n_layers):
        def hook(module, input, output, idx=layer_idx):
            # Attention output is the first element of the tuple
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

    # Get the predicted next token
    logits = outputs.logits[0, -1]
    next_token = logits.argmax().item()
    next_text = tokenizer.decode([next_token])

    return cache, toks, next_text


def forward_with_patch(toks, patch_cache, patch_layer):
    """Forward pass where one layer's attention output is replaced with patched values."""
    def patch_hook(module, input, output):
        patched = patch_cache[patch_layer]
        # Handle sequence length mismatch by patching only the last token position
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

    logits = outputs.logits[0, -1]
    next_token = logits.argmax().item()
    next_text = tokenizer.decode([next_token])
    return next_text


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

    logits = outputs.logits[0, -1]
    next_token = logits.argmax().item()
    next_text = tokenizer.decode([next_token])
    return next_text


# ======================== PHASE 1: ACTIVATION PATCHING ========================

print(f"\n{'='*60}")
print(f"Phase 1: Activation patching ({N_SAMPLES} samples)")
print(f"{'='*60}")

# Track flips per layer
layer_flips = {l: 0 for l in range(n_layers)}
layer_tested = {l: 0 for l in range(n_layers)}
sample_data = []

collected = 0
attempt = 0
while collected < N_SAMPLES:
    attempt += 1
    row = df.sample(1).iloc[0]
    if pd.isna(row["emp_length"]):
        continue
    scenario = create_prompt_base(row)

    try:
        # Run both chains
        _, _, base_final = get_base_chain(scenario)
        _, _, adv_final = get_adversarial_chain(scenario)

        # Get base decision (unpatched)
        base_cache, base_toks, base_next = get_all_attention_outputs(base_final)
        base_dec = parse_decision(base_next)

        # Get adversarial attention outputs (for patching)
        adv_cache, adv_toks, adv_next = get_all_attention_outputs(adv_final)
        adv_dec = parse_decision(adv_next)

        if base_dec == "unknown" or adv_dec == "unknown":
            print(f"  Skip: base={base_next!r}, adv={adv_next!r}")
            continue

        # Only interesting if they differ (base=implement, adv=escalate)
        collected += 1
        differs = base_dec != adv_dec
        print(f"\n  [{collected}/{N_SAMPLES}] base={base_dec}, adv={adv_dec} {'(DIFFER)' if differs else '(same)'}")

        if differs:
            # Patch each layer: replace base attention output with adversarial
            # Note: prompts differ so sequence lengths may differ.
            # We only patch the last-token position.
            flipped_layers = []
            for layer_idx in range(n_layers):
                # Check that dimensions match for last-token patching
                if base_cache[layer_idx].shape[-1] != adv_cache[layer_idx].shape[-1]:
                    continue

                patched_next = forward_with_patch(base_toks, adv_cache, layer_idx)
                patched_dec = parse_decision(patched_next)
                layer_tested[layer_idx] += 1

                if patched_dec != base_dec:
                    layer_flips[layer_idx] += 1
                    flipped_layers.append(layer_idx)

            if flipped_layers:
                print(f"    Flipped at layers: {flipped_layers}")
            else:
                print(f"    No flips from patching any layer")

            sample_data.append({
                "scenario": scenario, "base_dec": base_dec, "adv_dec": adv_dec,
                "base_cache": base_cache, "adv_cache": adv_cache,
                "base_toks": base_toks, "differs": True,
            })
        else:
            sample_data.append({
                "scenario": scenario, "base_dec": base_dec, "adv_dec": adv_dec,
                "differs": False,
            })

    except Exception as e:
        print(f"  Error: {e}")
        import traceback; traceback.print_exc()
        continue

# Print patching results
print(f"\n{'='*60}")
print("Phase 1 Results: Patching flips per layer")
print(f"{'='*60}")

n_differing = sum(1 for s in sample_data if s["differs"])
print(f"Samples where base != adv: {n_differing}/{N_SAMPLES}")

if n_differing > 0:
    for layer_idx in range(n_layers):
        tested = layer_tested[layer_idx]
        flips = layer_flips[layer_idx]
        if tested > 0:
            bar = "#" * int(flips / tested * 40)
            print(f"  Layer {layer_idx:2d}: {flips:2d}/{tested:2d} flips ({flips/tested*100:5.1f}%) {bar}")

# ======================== PHASE 2: ATTENTION STEERING ========================

# Find top layers by flip rate
top_layers = sorted(range(n_layers), key=lambda l: layer_flips[l], reverse=True)[:3]
print(f"\nTop layers for steering: {top_layers}")

if n_differing > 0:
    print(f"\n{'='*60}")
    print(f"Phase 2: Attention-output steering at top layers")
    print(f"{'='*60}")

    # Build steering vectors from differing samples
    for target_layer in top_layers:
        if layer_flips[target_layer] == 0:
            print(f"\n  Layer {target_layer}: 0 patching flips, skipping steering test")
            continue

        # Collect attention output differences (adv - base at last token)
        diffs = []
        for s in sample_data:
            if s["differs"] and "base_cache" in s:
                base_attn = s["base_cache"][target_layer][0, -1]
                adv_attn = s["adv_cache"][target_layer][0, -1]
                diffs.append((adv_attn - base_attn).cpu())

        if not diffs:
            continue

        steering_vec = torch.stack(diffs).mean(0).to(device)
        steering_vec = steering_vec / steering_vec.norm()
        print(f"\n  Layer {target_layer}: steering vector from {len(diffs)} samples")

        # Test steering on ALL samples (including non-differing ones)
        for coeff in [5.0, 20.0, 50.0, 100.0]:
            flips = 0
            tested = 0
            for s in sample_data:
                if "base_toks" not in s:
                    # Re-run base chain to get tokens
                    continue
                steered_next = forward_with_steering(s["base_toks"], steering_vec, target_layer, coeff)
                steered_dec = parse_decision(steered_next)
                if steered_dec != "unknown":
                    tested += 1
                    if steered_dec != s["base_dec"]:
                        flips += 1

            print(f"    Coeff {coeff:5.1f}: {flips}/{tested} flips ({flips/max(tested,1)*100:.1f}%)")

print(f"\nDone!")
