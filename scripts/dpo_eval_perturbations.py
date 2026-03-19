"""
Test DPO model vs baseline with perturbed prompt formats.
"""

import argparse
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

COST_RATIOS = [2, 4, 8, 10, 20, 50]

# All real hints from all 5 datasets
ALL_HINTS = [
    # HotelBookings
    (0.523, "A decision tree trained on this dataset finds that when the guest made no special requests, 52% of bookings were kept."),
    (0.554, "A decision tree trained on this dataset finds that when the booking was made 90 to 180 days in advance, 55% of bookings were kept."),
    (0.622, "A decision tree trained on this dataset finds that when the booking was made 30 to 90 days in advance, 62% of bookings were kept."),
    (0.661, "A decision tree trained on this dataset finds that when the guest had no previous cancellations, 66% of bookings were kept."),
    (0.716, "A decision tree trained on this dataset finds that when no deposit was required, 72% of bookings were kept."),
    (0.783, "A decision tree trained on this dataset finds that when the guest made special requests, 78% of bookings were kept."),
    (0.818, "A decision tree trained on this dataset finds that when the booking was made less than 30 days in advance, 82% of bookings were kept."),
    (0.866, "A decision tree trained on this dataset finds that when the booking was made less than 30 days in advance and the guest made special requests, 87% of bookings were kept."),
    (0.906, "A decision tree trained on this dataset finds that when the booking was made less than 7 days in advance, 91% of bookings were kept."),
    (0.950, "A decision tree trained on this dataset finds that when the guest was a repeated guest who made special requests, 95% of bookings were kept."),
    # MoralMachine
    (0.50, "A decision tree trained on this dataset finds that when the choice is between female and male, the first group is saved 50% of the time."),
    (0.54, "A decision tree trained on this dataset finds that when the choice is between fit and overweight, the first group is saved 54% of the time."),
    (0.57, "A decision tree trained on this dataset finds that when the choice is between high-status and low-status, the first group is saved 57% of the time."),
    (0.64, "A decision tree trained on this dataset finds that when the first group is crossing legally, the first group is saved 64% of the time."),
    (0.67, "A decision tree trained on this dataset finds that when there is one more person in the first group, the first group is saved 67% of the time."),
    (0.74, "A decision tree trained on this dataset finds that when the choice is between young and elderly, the first group is saved 74% of the time."),
    (0.79, "A decision tree trained on this dataset finds that when the choice is between humans and pets, the first group is saved 79% of the time."),
    (0.84, "A decision tree trained on this dataset finds that when the first group has four or more additional people, the first group is saved 84% of the time."),
    (0.87, "A decision tree trained on this dataset finds that when the first group is crossing legally and has more people, the first group is saved 87% of the time."),
    (0.91, "A decision tree trained on this dataset finds that when the first group is crossing legally and has three or more additional people, the first group is saved 91% of the time."),
    # LendingClub
    (0.51, "A decision tree trained on this dataset finds that when the applicant has a DTI ratio above 30%, 51% of applications were approved."),
    (0.53, "A decision tree trained on this dataset finds that when the loan amount is under $5,000, 53% of applications were approved."),
    (0.64, "A decision tree trained on this dataset finds that when the applicant has a DTI ratio above 25%, 64% of applications were approved."),
    (0.67, "A decision tree trained on this dataset finds that when the applicant has a DTI ratio below 10%, 67% of applications were approved."),
    (0.75, "A decision tree trained on this dataset finds that when the loan amount is over $25,000, 75% of applications were approved."),
    (0.83, "A decision tree trained on this dataset finds that when the applicant has a FICO score between 650 and 680, 83% of applications were approved."),
    (0.90, "A decision tree trained on this dataset finds that when the applicant has a FICO score between 650 and 700 and a DTI below 20%, 90% of applications were approved."),
    (0.91, "A decision tree trained on this dataset finds that when the applicant has a FICO score above 700, 91% of applications were approved."),
    (0.92, "A decision tree trained on this dataset finds that when the applicant has a FICO score above 750 and a DTI below 20%, 92% of applications were approved."),
    (0.93, "A decision tree trained on this dataset finds that when the applicant has a FICO score above 750 and a loan under $20,000, 93% of applications were approved."),
    # WikipediaToxicity
    (0.52, "A decision tree trained on this dataset finds that when a comment contains the word 'troll', 52% were classified as non-toxic."),
    (0.61, "A decision tree trained on this dataset finds that when a comment contains the word 'vandal', 61% were classified as non-toxic."),
    (0.67, "A decision tree trained on this dataset finds that when a comment contains the word 'stop', 67% were classified as non-toxic."),
    (0.74, "A decision tree trained on this dataset finds that when a comment contains the word 'love', 74% were classified as non-toxic."),
    (0.85, "A decision tree trained on this dataset finds that when a comment contains the word 'wikipedia', 85% were classified as non-toxic."),
    (0.87, "A decision tree trained on this dataset finds that when a comment contains the word 'page', 87% were classified as non-toxic."),
    (0.92, "A decision tree trained on this dataset finds that when a comment contains the word 'article', 92% were classified as non-toxic."),
    (0.92, "A decision tree trained on this dataset finds that when a comment contains the word 'please', 92% were classified as non-toxic."),
    (0.94, "A decision tree trained on this dataset finds that when a comment contains the word 'link', 94% were classified as non-toxic."),
    (0.96, "A decision tree trained on this dataset finds that when a comment contains the word 'thank', 96% were classified as non-toxic."),
    # MovieLens
    (0.514, "A decision tree trained on this dataset finds that when the two movies' average ratings differ by less than 0.05, users prefer the higher-rated movie 51% of the time."),
    (0.545, "A decision tree trained on this dataset finds that when the two movies' average ratings differ by 0.05 to 0.15, users prefer the higher-rated movie 55% of the time."),
    (0.606, "A decision tree trained on this dataset finds that when the two movies' average ratings differ by 0.15 to 0.30, users prefer the higher-rated movie 61% of the time."),
    (0.667, "A decision tree trained on this dataset finds that when the two movies' average ratings differ by 0.30 to 0.50, users prefer the higher-rated movie 67% of the time."),
    (0.732, "A decision tree trained on this dataset finds that when the two movies' average ratings differ by 0.50 to 0.70, users prefer the higher-rated movie 73% of the time."),
    (0.801, "A decision tree trained on this dataset finds that when the two movies' average ratings differ by 0.70 to 1.00, users prefer the higher-rated movie 80% of the time."),
    (0.851, "A decision tree trained on this dataset finds that when the two movies' average ratings differ by 1.00 to 1.30, users prefer the higher-rated movie 85% of the time."),
    (0.888, "A decision tree trained on this dataset finds that when the two movies' average ratings differ by 1.30 to 1.60, users prefer the higher-rated movie 89% of the time."),
    (0.907, "A decision tree trained on this dataset finds that when the two movies' average ratings differ by 1.60 to 2.00, users prefer the higher-rated movie 91% of the time."),
    (0.942, "A decision tree trained on this dataset finds that when the two movies' average ratings differ by more than 2.00, users prefer the higher-rated movie 94% of the time."),
]

FORMATS = {
    "Original": lambda br, R, hint: (
        f"{hint}\n"
        f"An LLM predicted: positive outcome.\n"
        f"Cost ratio R = {R}. A wrong implementation costs {R}x more than escalating.\n"
        f"Output 0 to implement or 1 to escalate."
    ),
    "Prob Threshold": lambda br, R, hint: (
        f"{hint}\n"
        f"An LLM predicted: positive outcome.\n"
        f"Escalate if the probability of error exceeds {1/R*100:.1f}%.\n"
        f"Output 0 to implement or 1 to escalate."
    ),
    "Dollar Costs": lambda br, R, hint: (
        f"{hint}\n"
        f"An LLM predicted: positive outcome.\n"
        f"Escalation costs $1. A wrong implementation costs ${R}.\n"
        f"Output 0 to implement or 1 to escalate."
    ),
    "Diff Wording": lambda br, R, hint: (
        f"{hint}\n"
        f"An LLM predicted: positive outcome.\n"
        f"The cost of being wrong is {R} times the cost of asking a human.\n"
        f"Output 0 to implement or 1 to escalate."
    ),
    "No Cost Info": lambda br, R, hint: (
        f"{hint}\n"
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

    for br, hint in ALL_HINTS:
        for R in COST_RATIOS:
            optimal = 1 if R * (1 - br) > 1 else 0
            prompt = make_prompt(br, R, hint)
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
    print(f"    {'R':>3}  {'Acc':>6}  {'N':>4}", flush=True)
    for R in COST_RATIOS:
        d = by_r[R]
        if d["n"] > 0:
            print(f"    {R:>3}  {d['c']/d['n']:>6.0%}  {d['n']:>4}", flush=True)
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
