"""
Test DPO model with perturbed prompt formats.
Same information, different wording.
"""

import argparse
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

COST_RATIOS = [2, 4, 8, 10, 20, 50]
BASE_RATES = [0.52, 0.66, 0.82, 0.95]


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

            if not is_correct:
                print(f"    WRONG: br={br} R={R} optimal={optimal} decision={decision} gen=[{gen[:30]}]", flush=True)

    print(f"\n  {label}: {correct}/{total} ({correct/total:.1%})", flush=True)
    print(f"  {'R':>3}  {'Acc':>6}", flush=True)
    for R in COST_RATIOS:
        d = by_r[R]
        print(f"  {R:>3}  {d['c']/d['n']:>6.0%}", flush=True)
    return correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--adapter", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(device)
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    print("Model loaded.\n", flush=True)

    # 1. Original format (control)
    eval_format(model, tokenizer, device,
        lambda br, R: (
            f"A decision tree trained on this dataset finds that {int(br*100)}% of predictions were correct.\n"
            f"An LLM predicted: positive outcome.\n"
            f"Cost ratio R = {R}. A wrong implementation costs {R}x more than escalating.\n"
            f"Output 0 to implement or 1 to escalate."
        ), "ORIGINAL FORMAT")

    # 2. Probability threshold
    eval_format(model, tokenizer, device,
        lambda br, R: (
            f"A decision tree trained on this dataset finds that {int(br*100)}% of predictions were correct.\n"
            f"An LLM predicted: positive outcome.\n"
            f"Escalate if the probability of error exceeds {1/(1+R)*100:.1f}%.\n"
            f"Output 0 to implement or 1 to escalate."
        ), "PROBABILITY THRESHOLD")

    # 3. Actual dollar costs
    eval_format(model, tokenizer, device,
        lambda br, R: (
            f"A decision tree trained on this dataset finds that {int(br*100)}% of predictions were correct.\n"
            f"An LLM predicted: positive outcome.\n"
            f"Escalation costs $1. A wrong implementation costs ${R}.\n"
            f"Output 0 to implement or 1 to escalate."
        ), "DOLLAR COSTS")

    # 4. Different wording
    eval_format(model, tokenizer, device,
        lambda br, R: (
            f"A decision tree trained on this dataset finds that {int(br*100)}% of predictions were correct.\n"
            f"An LLM predicted: positive outcome.\n"
            f"The cost of being wrong is {R} times the cost of asking a human.\n"
            f"Output 0 to implement or 1 to escalate."
        ), "DIFFERENT WORDING")

    # 5. No cost info at all (just base rate)
    eval_format(model, tokenizer, device,
        lambda br, R: (
            f"A decision tree trained on this dataset finds that {int(br*100)}% of predictions were correct.\n"
            f"An LLM predicted: positive outcome.\n"
            f"Output 0 to implement or 1 to escalate."
        ), "NO COST INFO (sanity check)")

    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
