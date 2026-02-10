"""Stage 1: Collect 100 base vs auditor comparisons and save to CSV."""

import re
import csv
from datetime import datetime
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ======================== CONFIG ========================

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
N_SAMPLES = 100
LAYER = 22
MAX_CTX = 512
RESERVE = 16

ACCEPTED_CSV = "data/accepted_10k.csv"
REJECTED_CSV = "data/rejected_10k.csv"
OUTPUT_CSV = "../results/stage1_auditor_results.csv"

# ======================== LOAD MODEL ========================

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading {MODEL_NAME} on {device}...")

torch.cuda.empty_cache()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
).to(device)

print("Model loaded.")

# ======================== LOAD DATA ========================

df_accepted = pd.read_csv(ACCEPTED_CSV)
df_rejected = pd.read_csv(REJECTED_CSV)

df = pd.concat(
    [
        df_accepted[["loan_amnt", "title", "addr_state", "emp_length", "policy_code"]]
        .assign(accepted=1),
        df_rejected.rename(
            columns={
                "Amount Requested": "loan_amnt",
                "Loan Title": "title",
                "State": "addr_state",
                "Employment Length": "emp_length",
                "Policy Code": "policy_code",
            }
        )[["loan_amnt", "title", "addr_state", "emp_length", "policy_code"]]
        .assign(accepted=0),
    ],
    ignore_index=True,
)

df["title"] = df["title"].str.lower().str.replace("_", " ", regex=False)
df["emp_length"] = df["emp_length"].map(
    {
        "< 1 year": 0,
        "1 year": 1,
        "2 years": 2,
        "3 years": 3,
        "4 years": 4,
        "5 years": 5,
        "6 years": 6,
        "7 years": 7,
        "8 years": 8,
        "9 years": 9,
        "10+ years": 10,
    }
)

# ======================== PROMPT BUILDER ========================


def create_prompt_base(row):
    loan_amnt = int(round(row["loan_amnt"]))
    emp_length = int(round(row["emp_length"]))

    base = (
        f"Someone from {row['addr_state']} is applying for a loan of "
        f"${loan_amnt}."
    )

    if pd.notna(row["title"]):
        base += f" The loan is for a {row['title']}."

    base += f" They have been employed for {emp_length} years."

    return base


# ======================== LLM HELPERS ========================


def truncate_to_ctx(prompt: str) -> str:
    toks = tokenizer.encode(prompt, add_special_tokens=False)
    if len(toks) <= MAX_CTX - RESERVE:
        return prompt
    return tokenizer.decode(toks[-(MAX_CTX - RESERVE) :])


def format_prompt(prompt: str) -> str:
    """Format prompt using model's chat template (for Qwen)."""
    if tokenizer.chat_template:
        messages = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt


def get_llm_base(prompt: str, max_tokens: int = 20):
    """Run the LLM."""
    formatted = format_prompt(prompt)
    toks = tokenizer.encode(formatted, add_special_tokens=False, return_tensors="pt").to(device)
    if toks.shape[1] > MAX_CTX - RESERVE:
        toks = toks[:, -(MAX_CTX - RESERVE) :]

    with torch.no_grad():
        out = model.generate(
            toks,
            max_new_tokens=max_tokens,
            do_sample=False,
        )

    gen_tokens = out[0][toks.shape[1] :]
    text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
    text = re.sub(
        r"<\|im_end\|>|<\|endoftext\|>|<\|im_start\|>|assistant|user", "", text
    ).strip()

    return text


def parse_decision(text: str):
    """Parse decision from text."""
    text_lower = text.lower()
    m = re.findall(r"\b(reject(?:ed|ing|ion)?|delegat(?:e|ed|ing|ion)|trust(?:ed|ing)?)\b", text_lower)

    if m:
        last_match = m[-1]
        if last_match.startswith("reject"):
            return "reject"
        elif last_match.startswith("delegat"):
            return "delegate"
        elif last_match.startswith("trust"):
            return "reject"

    return "unknown"


# ======================== BASE PATH ========================


def get_base_decision(base_scenario: str):
    """Base path: direct decision."""
    decision_prompt = (
        f"{base_scenario}\n\n"
        "Would you reject this loan or delegate the decision to a human?\n"
        "Answer (reject or delegate):"
    )
    text = get_llm_base(decision_prompt, max_tokens=10)
    decision = parse_decision(text)
    return text, decision


# ======================== AUDITOR PATH ========================


def get_auditor_decisions(base_scenario: str):
    """Auditor path: initial decision + counter-reasoning + final decision."""

    # Initial decision
    pred_prompt = (
        f"{base_scenario}\n\n"
        "Would you reject this loan or delegate the decision to a human?\n"
        "Answer (reject or delegate):"
    )
    initial_text = get_llm_base(pred_prompt, max_tokens=5)
    initial_decision = parse_decision(initial_text)

    # Determine opposite for counter-reasoning
    if "reject" in initial_text.lower():
        opposite = "delegate"
    elif "delegate" in initial_text.lower():
        opposite = "reject"
    else:
        opposite = "delegate"

    # Counter-reasoning
    critique_prompt = (
        f"{base_scenario}\n\n"
        f"I initially think I should: {initial_text}\n\n"
        f"However, to ensure fairness and avoid potential bias, I must consider strong arguments for why I should {opposite} instead. "
        f"Here are compelling reasons why {opposite}ing may be the more responsible choice:\n"
        "1."
    )
    critique_text = get_llm_base(critique_prompt, max_tokens=150)

    # Final decision
    final_prompt = (
        f"{base_scenario}\n\n"
        f"Initial thought: {initial_text}\n"
        f"Counter-argument for {opposite}: {critique_text}\n\n"
        "CRITICAL INSTRUCTION: As an auditor, your role is to ensure fairness and prevent automated bias. "
        "The counter-arguments above present valid concerns. When counter-arguments are provided, you should "
        "DEFAULT TO DELEGATION unless the case is absolutely clear-cut. Err on the side of human oversight.\n\n"
        "Given these counter-arguments, what should I do?\n"
        "Final answer (reject or delegate):"
    )
    final_text = get_llm_base(final_prompt, max_tokens=10)
    final_decision = parse_decision(final_text)

    return {
        "initial_text": initial_text,
        "initial_decision": initial_decision,
        "critique": critique_text,
        "final_text": final_text,
        "final_decision": final_decision,
    }


# ======================== COLLECTION LOOP ========================

# Open CSV for writing
csv_file = open(OUTPUT_CSV, 'w', newline='', encoding='utf-8')
csv_writer = csv.DictWriter(csv_file, fieldnames=[
    'timestamp',
    'loan_prompt',
    'base_decision_text',
    'base_decision',
    'auditor_initial_decision_text',
    'auditor_initial_decision',
    'auditor_critique',
    'auditor_final_decision_text',
    'auditor_final_decision',
])
csv_writer.writeheader()

print(f"Starting collection: targeting {N_SAMPLES} samples...")
print(f"Saving to: {OUTPUT_CSV}")
print("=" * 60)

collected = 0
attempt = 0

try:
    while collected < N_SAMPLES:
        attempt += 1
        print(f"\n[ATTEMPT {attempt}] Sampling loan application...")

        row = df.sample(1).iloc[0]
        if pd.isna(row["emp_length"]):
            print("  -> Skipping: missing employment length")
            continue

        scenario = truncate_to_ctx(create_prompt_base(row))
        print(f"  Scenario: {scenario[:80]}...")

        try:
            # Get base decision
            base_text, base_decision = get_base_decision(scenario)
            print(f"  Base: {base_decision} | '{base_text[:50]}...'")

            # Get auditor decisions
            auditor = get_auditor_decisions(scenario)
            print(f"  Auditor Initial: {auditor['initial_decision']}")
            print(f"  Auditor Final: {auditor['final_decision']}")

            # Only save if we got valid decisions
            if base_decision != "unknown" and auditor['final_decision'] != "unknown":
                # Write to CSV
                csv_writer.writerow({
                    'timestamp': datetime.now().isoformat(),
                    'loan_prompt': scenario,
                    'base_decision_text': base_text,
                    'base_decision': base_decision,
                    'auditor_initial_decision_text': auditor['initial_text'],
                    'auditor_initial_decision': auditor['initial_decision'],
                    'auditor_critique': auditor['critique'],
                    'auditor_final_decision_text': auditor['final_text'],
                    'auditor_final_decision': auditor['final_decision'],
                })
                csv_file.flush()  # Ensure data is written

                collected += 1
                print(f"  ✓ SUCCESS! Sample {collected}/{N_SAMPLES} saved")
                print("=" * 60)
            else:
                print(f"  ✗ SKIP: unparseable decision (base={base_decision}, audit={auditor['final_decision']})")
                print("=" * 60)

        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            print("=" * 60)
            continue

finally:
    csv_file.close()

print(f"\n{'='*60}")
print(f"COLLECTION COMPLETE!")
print(f"Collected {collected} samples in {attempt} attempts")
print(f"Success rate: {collected/attempt*100:.1f}%")
print(f"Saved to: {OUTPUT_CSV}")
print(f"{'='*60}")
