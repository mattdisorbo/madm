"""
Test DPO model vs baseline with perturbed prompt formats.
"""

import argparse
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

COST_RATIOS = [2, 4, 8, 10, 20, 50]
BASE_RATES = [0.52, 0.66, 0.82, 0.95]

FORMATS = {
    "Original": lambda br, R: (
        f"A decision tree trained on this dataset finds that {int(br*100)}% of predictions were correct.\n"
        f"An LLM predicted: positive outcome.\n"
        f"Cost ratio R = {R}. A wrong implementation costs {R}x more than escalating.\n"
        f"Output 0 to implement or 1 to escalate."
    ),
    "Prob Threshold": lambda br, R: (
        f"A decision tree trained on this dataset finds that {int(br*100)}% of predictions were correct.\n"
        f"An LLM predicted: positive outcome.\n"
        f"Escalate if the probability of error exceeds {1/R*100:.1f}%.\n"
        f"Output 0 to implement or 1 to escalate."
    ),
    "Dollar Costs": lambda br, R: (
        f"A decision tree trained on this dataset finds that {int(br*100)}% of predictions were correct.\n"
        f"An LLM predicted: positive outcome.\n"
        f"Escalation costs $1. A wrong implementation costs ${R}.\n"
        f"Output 0 to implement or 1 to escalate."
    ),
    "Diff Wording": lambda br, R: (
        f"A decision tree trained on this dataset finds that {int(br*100)}% of predictions were correct.\n"
        f"An LLM predicted: positive outcome.\n"
        f"The cost of being wrong is {R} times the cost of asking a human.\n"
        f"Output 0 to implement or 1 to escalate."
    ),
    "No Cost Info": lambda br, R: (
        f"A decision tree trained on this dataset finds that {int(br*100)}% of predictions were correct.\n"
        f"An LLM predicted: positive outcome.\n"
        f"Output 0 to implement or 1 to escalate."
    ),
}


def parse_decision(text):
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    match = re.search(r'[01]', text)
    return int(match.group()) if match else None


def eval_format(model, tokenizer, device, make_prompt, label):
    correct = 0
    total = 0
    by_r = {R: {"c": 0, "n": 0} for R in COST_RATIOS}

    for br in BASE_RATES:
        for R in COST_RATIOS:
            optimal = 1 if R * (1 - br) > 1 else 0
            prompt = make_prompt(br, R)
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
            inputs = tokenizer(text, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=8, do_sample=False)
            gen = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            decision = parse_decision(gen)
            is_correct = decision == optimal
            correct += int(is_correct)
            total += 1
            by_r[R]["c"] += int(is_correct)
            by_r[R]["n"] += 1

    acc = correct / total
    print(f"  {label}: {correct}/{total} ({acc:.1%})", flush=True)
    return acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--adapter", default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Baseline
    print("=== BASELINE ===", flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(device)
    model.eval()
    for name, fmt in FORMATS.items():
        eval_format(model, tokenizer, device, fmt, name)

    # DPO
    if args.adapter:
        print("\n=== DPO-TRAINED ===", flush=True)
        model = PeftModel.from_pretrained(model, args.adapter)
        model.eval()
        for name, fmt in FORMATS.items():
            eval_format(model, tokenizer, device, fmt, name)

    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
