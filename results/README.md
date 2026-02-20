# Neural Activation Auditor - Experiment Results

This folder contains the results from two stages of the neural activation auditor experiment.

## Overview

We're testing whether we can extract "delegation behavior" from a model that's been prompted to audit itself, and apply it to the base model without explicit prompting.

## Running the Experiments

### Stage 1: Base vs Auditor (100 samples)

Collects 100 loan application decisions comparing:
- **Base model**: Direct decision without critique
- **Auditor model**: Initial decision → counter-reasoning → final decision

**Run:**
```bash
cd /home/user/madm/madm-main
uv run python scripts/collect_stage1_auditor.py
```

**Output:** `stage1_auditor_results.csv`

**Columns:**
- `timestamp` - When the sample was collected
- `loan_prompt` - The actual loan application scenario
- `base_decision_text` - Raw text output from base model
- `base_decision` - Parsed decision (reject/delegate/unknown)
- `auditor_initial_decision_text` - Raw text of auditor's first decision
- `auditor_initial_decision` - Parsed initial decision
- `auditor_critique` - The counter-reasoning text
- `auditor_final_decision_text` - Raw text of auditor's final decision
- `auditor_final_decision` - Parsed final decision

**Expected runtime:** ~20-30 minutes

---

### Stage 2: Base vs Steered (100 samples)

Collects 100 loan application decisions comparing:
- **Base model**: Direct decision
- **Steered model**: Base model with activation steering applied

This script will:
1. First collect 30 samples to train an SAE
2. Extract the steering vector
3. Then collect 100 samples comparing base vs steered

**Run:**
```bash
cd /home/user/madm/madm-main
uv run python scripts/collect_stage2_steering.py
```

**Output:** `stage2_steering_results.csv`

**Columns:**
- `timestamp` - When the sample was collected
- `loan_prompt` - The actual loan application scenario
- `base_decision_text` - Raw text output from base model
- `base_decision` - Parsed decision (reject/delegate/unknown)
- `steered_decision_text` - Raw text with steering applied
- `steered_decision` - Parsed steered decision

**Expected runtime:** ~25-35 minutes

---

## What We're Testing

### Hypothesis
1. Base model has a rejection bias
2. Heavy-handed auditing (with critique) can flip decisions to "delegate"
3. We can extract the neural "essence" of delegation via activation differences
4. Applying that essence directly (via steering) mimics auditing without explicit prompting

### Expected Results

**Stage 1:** We expect to see cases where:
- Base model says "reject"
- Auditor initially says "reject"
- Auditor (after critique) changes to "delegate"

**Stage 2:** We expect to see cases where:
- Base model says "reject"
- Steered model (without any critique prompt) says "delegate"

This would confirm that steering captures the "delegation mindset" mechanistically!

---

## Analysis Ideas

After running both stages, you can analyze:

1. **Flip rates:**
   - Stage 1: How often does auditor change decisions?
   - Stage 2: How often does steering change decisions?

2. **Comparison:**
   - Are the flip rates similar?
   - Does steering achieve similar behavior to auditing?

3. **Text quality:**
   - Are steered outputs coherent?
   - Do they use similar language to audited outputs?

4. **Efficiency:**
   - Steering is a single forward pass
   - Auditing requires 3 LLM calls (initial, critique, final)
   - How much computational savings do we get?
