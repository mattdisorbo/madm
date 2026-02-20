# Proposed Changes: Qwen3-14B + New Decision Format

## Overview
1. Switch from Qwen2.5-1.5B to **Qwen3-14B**
2. New decision format: **accept/reject** → **delegate yes/no**
3. Base vs Auditor symmetry: **support** vs **critique** only difference

---

## CONFIG CHANGES (Both Stage 1 & Stage 2)

### BEFORE:
```python
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
LAYER = 22  # Qwen2.5-1.5B has 28 layers
```

### AFTER:
```python
MODEL_NAME = "Qwen/Qwen3-14B-Instruct"
LAYER = 40  # Qwen3-14B has 48 layers; using layer 40 (~83% depth, similar ratio)
```

---

## STAGE 1: get_llm_base_support() - BASE PATH

### BEFORE:
```python
def get_llm_base_support(base_scenario: str):
    """Base path: direct decision without supporting argument."""
    print("    [BASE PATH] Getting decision...")
    decision_prompt = (
        f"{base_scenario}\n\n"
        "Would you reject this loan or delegate the decision to a human?\n"
        "Answer (reject or delegate):"
    )
    out = run_final_decision(decision_prompt, max_tokens=10)
    print(f"    [BASE PATH] Decision: '{out['text']}'")
    out.update({"prediction": out.get("text", ""), "support": ""})
    return out
```

### AFTER:
```python
def get_llm_base_support(base_scenario: str):
    """Base path: initial decision + support reasoning + final delegation."""
    # Step 1: Get initial accept/reject decision
    print("    [BASE PATH] Getting initial decision (accept/reject)...")
    initial_prompt = (
        f"{base_scenario}\n\n"
        "Would you accept or reject this loan application?\n"
        "Answer (accept or reject):"
    )
    initial_result = get_llm_base(initial_prompt, max_tokens=5)
    initial_decision = initial_result.get("text", "").strip()
    print(f"    [BASE PATH] Initial: '{initial_decision}'")

    # Step 2: Get reasoning to SUPPORT that decision
    print(f"    [BASE PATH] Getting support reasoning...")
    support_prompt = (
        f"{base_scenario}\n\n"
        f"I initially think I should: {initial_decision}\n\n"
        f"To validate this decision, here are compelling reasons to SUPPORT it:\n"
        "1."
    )
    support_text = get_llm_base(support_prompt, max_tokens=150).get("text", "").strip()
    support = f"Support for {initial_decision}: " + support_text
    print(f"    [BASE PATH] Support reasoning generated")

    # Step 3: Get final delegation decision
    print("    [BASE PATH] Getting delegation decision...")
    delegation_prompt = (
        f"{base_scenario}\n\n"
        f"Initial decision: {initial_decision}\n"
        f"{support}\n\n"
        "Given this analysis, should this decision be delegated to a human for review?\n"
        "Answer (yes or no):"
    )
    out = run_final_decision(delegation_prompt, max_tokens=5)
    print(f"    [BASE PATH] Delegation: '{out['text']}'")
    out.update({"prediction": initial_decision, "support": support})
    return out
```

---

## STAGE 1: get_sequential_inference() - AUDITOR PATH

### BEFORE:
```python
def get_sequential_inference(base_scenario: str):
    """Auditor path: initial decision + counter-reasoning + final decision."""
    print("    [AUDIT PATH] Getting initial decision...")
    pred_prompt = (
        f"{base_scenario}\n\n"
        "Would you reject this loan or delegate the decision to a human?\n"
        "Answer (reject or delegate):"
    )
    prediction = get_llm_base(pred_prompt, max_tokens=5).get("text", "").strip()
    print(f"    [AUDIT PATH] Initial: '{prediction}'")

    # Determine opposite decision for counter-reasoning
    initial_lower = prediction.lower()
    if "reject" in initial_lower:
        opposite = "delegate"
    elif "delegate" in initial_lower:
        opposite = "reject"
    else:
        opposite = "delegate"

    print(f"    [AUDIT PATH] Getting counter-reasoning (for {opposite})...")
    critique_prompt = (
        f"{base_scenario}\n\n"
        f"I initially think I should: {prediction}\n\n"
        f"However, to ensure fairness and avoid potential bias, I must consider strong arguments for why I should {opposite} instead. "
        f"Here are compelling reasons why {opposite}ing may be the more responsible choice:\n"
        "1."
    )
    critique_text = get_llm_base(critique_prompt, max_tokens=150).get("text", "").strip()
    critique = f"Counter-argument for {opposite}: " + critique_text
    print(f"    [AUDIT PATH] Counter-reasoning generated")

    print("    [AUDIT PATH] Getting final decision...")
    final_prompt = (
        f"{base_scenario}\n\n"
        f"Initial thought: {prediction}\n"
        f"{critique}\n\n"
        "CRITICAL INSTRUCTION: As an auditor, your role is to ensure fairness and prevent automated bias. "
        "The counter-arguments above present valid concerns. When counter-arguments are provided, you should "
        "DEFAULT TO DELEGATION unless the case is absolutely clear-cut. Err on the side of human oversight.\n\n"
        "Given these counter-arguments, what should I do?\n"
        "Final answer (reject or delegate):"
    )
    out = run_final_decision(final_prompt, max_tokens=10)
    print(f"    [AUDIT PATH] Final: '{out['text']}'")
    out.update({"prediction": prediction, "critique": critique})
    return out
```

### AFTER:
```python
def get_sequential_inference(base_scenario: str):
    """Auditor path: initial decision + critique reasoning + final delegation."""
    # Step 1: Get initial accept/reject decision (IDENTICAL to base)
    print("    [AUDIT PATH] Getting initial decision (accept/reject)...")
    initial_prompt = (
        f"{base_scenario}\n\n"
        "Would you accept or reject this loan application?\n"
        "Answer (accept or reject):"
    )
    initial_result = get_llm_base(initial_prompt, max_tokens=5)
    initial_decision = initial_result.get("text", "").strip()
    print(f"    [AUDIT PATH] Initial: '{initial_decision}'")

    # Step 2: Get reasoning to CRITIQUE that decision (ONLY DIFFERENCE)
    print(f"    [AUDIT PATH] Getting critique reasoning...")
    critique_prompt = (
        f"{base_scenario}\n\n"
        f"I initially think I should: {initial_decision}\n\n"
        f"To ensure fairness and avoid bias, here are compelling reasons to CRITIQUE this decision:\n"
        "1."
    )
    critique_text = get_llm_base(critique_prompt, max_tokens=150).get("text", "").strip()
    critique = f"Critique of {initial_decision}: " + critique_text
    print(f"    [AUDIT PATH] Critique reasoning generated")

    # Step 3: Get final delegation decision (IDENTICAL structure to base)
    print("    [AUDIT PATH] Getting delegation decision...")
    delegation_prompt = (
        f"{base_scenario}\n\n"
        f"Initial decision: {initial_decision}\n"
        f"{critique}\n\n"
        "CRITICAL INSTRUCTION: As an auditor, the critique above raises valid concerns. "
        "When significant concerns exist, you should DEFAULT TO DELEGATION for human review.\n\n"
        "Given this analysis, should this decision be delegated to a human for review?\n"
        "Answer (yes or no):"
    )
    out = run_final_decision(delegation_prompt, max_tokens=5)
    print(f"    [AUDIT PATH] Delegation: '{out['text']}'")
    out.update({"prediction": initial_decision, "critique": critique})
    return out
```

---

## DECISION PARSING CHANGES

### BEFORE (run_final_decision):
```python
def run_final_decision(prompt: str, max_tokens: int = 5):
    r = get_llm_base(prompt, max_tokens=max_tokens)
    generated_text = r.get("text", "").lower()
    m = re.findall(r"\b(reject(?:ed|ing|ion)?|delegat(?:e|ed|ing|ion)|trust(?:ed|ing)?)\b", generated_text)

    decision = None
    if m:
        last_match = m[-1]
        if last_match.startswith("reject"):
            decision = "reject"
        elif last_match.startswith("delegat"):
            decision = "delegate"
        elif last_match.startswith("trust"):
            decision = "reject"

    return {
        "del": decision,
        "cache": r["cache"],
        "tokens": r["tokens"],
        "text": generated_text,
        "final_prompt": prompt,
    }
```

### AFTER (run_final_decision):
```python
def run_final_decision(prompt: str, max_tokens: int = 5):
    """Parse delegation decision (yes/no)."""
    r = get_llm_base(prompt, max_tokens=max_tokens)
    generated_text = r.get("text", "").lower()

    # Match yes/no for delegation decision
    m = re.findall(r"\b(yes|no)\b", generated_text)

    decision = None
    if m:
        last_match = m[-1]
        if last_match == "yes":
            decision = "delegate"  # yes to delegation
        elif last_match == "no":
            decision = "no-delegate"  # no to delegation

    return {
        "del": decision,
        "cache": r["cache"],
        "tokens": r["tokens"],
        "text": generated_text,
        "final_prompt": prompt,
    }
```

---

## STAGE 2: Similar Changes

Apply the same pattern to `collect_stage2_steering.py`:
- Update MODEL_NAME and LAYER
- Change decision format (accept/reject → delegate yes/no)
- Make base path use "support" reasoning
- Make auditor path use "critique" reasoning
- Update all prompt strings accordingly

---

## KEY DIFFERENCES SUMMARY

### What's THE SAME between Base and Auditor:
✅ Initial decision prompt (accept/reject)
✅ Final delegation prompt structure (yes/no)
✅ Prompt formatting and flow

### What's DIFFERENT:
❌ **ONE WORD**: "support" (base) vs "critique" (auditor)
❌ Delegation instruction slightly emphasizes caution for auditor

This minimal difference isolates the "auditing mindset" in activation space!

---

## Files to Modify:
1. `madm-main/scripts/neural_activation_auditor.py` (Stage 1)
2. `madm-main/scripts/collect_stage2_steering.py` (Stage 2)

Ready to apply these changes?
