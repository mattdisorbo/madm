# SFT Fine-Tuning for Cost-Sensitive Escalation (Qwen3-8B)

## Goal

Train a model to learn when to escalate a prediction to a human, given the cost ratio between escalation labor cost and wrong-answer cost. The optimal policy escalates when P(wrong) > 1/(1+R), where R = c_w/c_l is the cost ratio. Oracle labels are derived directly from the hint's stated base rate rather than from empirical accuracy.

## Model

- Predictions: Qwen3.5-9B (Call 1 predictions via Together API)
- Fine-tuned: Qwen3-8B (closest fine-tunable model, LoRA via Together AI, $7.70)
- Hyperparameters: 3 epochs, lr=1e-5, batch_size=max, train_on_inputs=False
- Training loss: 0.51 (epoch 1) -> 0.06 (epoch 2) -> 0.05 (epoch 3)

## Data

Multi-turn format matching the study3 pipeline:
1. User: scenario + hint + predict prompt
2. Assistant: model's prediction (frozen Call 1 output from Qwen3.5-9B)
3. User: cost framing + escalation prompt
4. Assistant: oracle label ("0" implement or "1" escalate)

Oracle labels use the hint's base rate directly: escalate if (1 - base_rate) > 1/(1+R).

- **Training set** (`train.jsonl`): 12,000 examples (10 hints x 6 cost ratios x 200 samples)
- **Holdout set** (`holdout.jsonl`): 2,832 examples (10 hints x 6 cost ratios x ~47 samples)
- **6 cost ratios**: R = 2, 4, 8, 10, 20, 50

## Results

**Not evaluated.** Together AI's dedicated endpoint for the fine-tuned model (harang/Qwen3-8B-hotel-escalation-9a69e372) failed to serve requests despite reporting STARTED status. The Qwen3-8B base model is not available for serverless inference of fine-tuned variants.

The Qwen2.5-7B fine-tune (see `results/fine-tuning-qwen2.5-7b/`) did evaluate and showed that SFT does not learn the cost-sensitive policy: the model escalates at a roughly constant ~20% rate regardless of cost ratio.

## Script

See `scripts/together_finetune_qwen35.py` for data preparation, training, and evaluation code.
