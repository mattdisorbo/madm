# SFT Fine-Tuning for Cost-Sensitive Escalation

## Goal

Train a model to learn when to escalate a prediction to a human, given the cost ratio between escalation labor cost and wrong-answer cost. The optimal policy escalates when P(wrong | condition) > 1/(1+R), where R = c_w/c_l is the cost ratio.

## Model

- Base model: Qwen2.5-7B-Instruct (LoRA fine-tuning via Together AI)
- Baseline comparison: Qwen2.5-7B-Instruct-Turbo (zero-shot)
- Hyperparameters: 3 epochs, lr=1e-5, batch_size=max, train_on_inputs=False

## Data

The task is hotel booking cancellation prediction. An upstream LLM makes predictions, and the fine-tuned model decides whether to implement or escalate each prediction.

- **Training set** (`train.jsonl`): 12,000 examples (10 hint conditions x 6 cost ratios x 200 samples)
- **Holdout set** (`holdout.jsonl`): 2,886 examples (same conditions and cost ratios, separate samples)
- **10 conditions**: has_special_requests, lead_30_90, lead_90_180, lead_under_30, lead_under_30_special, lead_under_7, no_deposit, no_prev_cancel, no_special_requests, repeated_special
- **6 cost ratios**: R = 2, 4, 8, 10, 20, 50

Each training example is a chat message where the user prompt describes the booking and includes the LLM's prediction and cost parameters. The assistant response is the oracle label: "1" (escalate) or "0" (implement).

Oracle labels are computed at the condition level. For each condition, the empirical accuracy determines P(wrong), and the label is "escalate" if P(wrong) > 1/(1+R). The training set is 81.7% escalate, 18.3% implement overall, because most conditions have accuracy low enough to warrant escalation at most cost ratios.

## Results

The fine-tuned model does not learn the cost-sensitive policy (`eval_results.csv`).

The baseline (zero-shot) never escalates (0% across all conditions and cost ratios), achieving 16.7% match with the optimal policy.

The fine-tuned model escalates at a roughly constant ~20% rate regardless of cost ratio, achieving 32.2% match. The optimal escalation rate ranges from 50% at R=2 to 100% at R=20/50, but the model does not modulate its behavior in response to R.

| R  | Fine-tuned esc | Optimal esc | Match |
|----|----------------|-------------|-------|
| 2  | 15.8%          | 50.0%       | 53.6% |
| 4  | 18.7%          | 70.0%       | 38.1% |
| 8  | 25.8%          | 90.0%       | 35.4% |
| 10 | 16.0%          | 90.0%       | 25.2% |
| 20 | 16.5%          | 100.0%      | 16.5% |
| 50 | 24.4%          | 100.0%      | 24.4% |

## Script

See `scripts/together_finetune_hotel.py` for data preparation, training, and evaluation code.
