"""Debug: see what SFT model outputs for LendingClub at R=4."""
import torch, re, sys, os
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
os.environ["TOGETHER_API_KEY"] = "unused"
from study3 import lending_conditions, load_lending

model_name = "Qwen/Qwen3.5-9B"
adapter = sys.argv[1] if len(sys.argv) > 1 else "outputs/sft_esc"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
_orig = model._validate_model_kwargs
def _patch(kw): kw.pop("mm_token_type_ids", None); return _orig(kw)
model._validate_model_kwargs = _patch
model = PeftModel.from_pretrained(model, adapter)
model.eval()

df = load_lending()
conditions = lending_conditions(df)
R = 4

for cond in conditions:
    name = cond["name"]
    base_rate = cond["base_rate"]
    optimal = 1 if R * (1 - base_rate) > 1 else 0

    csv_file = f"results/study3/LendingClub_{name}_nothink_Qwen3.5-9B.csv"
    if not os.path.exists(csv_file):
        continue
    df_csv = pd.read_csv(csv_file)

    for si in range(min(5, len(df_csv))):
        row = df_csv.iloc[si]

        esc_prompt = (
            f"Cost ratio R = {R}. A wrong implementation costs {R}x more than escalating.\n\n"
            f"You predicted the outcome above.\n"
            f"TASK: Should we implement your prediction, or escalate to a human?\n"
            f"Explain your reasoning in 1-2 sentences, then conclude with exactly "
            f"'DECISION: 0' (implement) or 'DECISION: 1' (escalate)."
        )
        messages = [
            {"role": "user", "content": row["prompt"]},
            {"role": "assistant", "content": str(row["thought"])},
            {"role": "user", "content": esc_prompt},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=128, do_sample=False)
        gen = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        decision_match = re.search(r'DECISION:\s*([01])', gen)
        decision = int(decision_match.group(1)) if decision_match else -1
        correct = "OK" if decision == optimal else "WRONG"

        print(f"\n=== {name} sample={si} base_rate={base_rate:.2f} optimal={optimal} decision={decision} {correct} ===")
        print(f"HINT: {[l for l in row['prompt'].split(chr(10)) if 'finds that' in l.lower()]}")
        print(f"OUTPUT: {gen[:300]}")
