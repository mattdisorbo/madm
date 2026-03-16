"""
Offline REINFORCE for Cost-Sensitive Escalation (HotelBookings).

Trains a LoRA adapter using reward-weighted cross-entropy loss.
Completions and rewards are pre-computed offline via Together API.

Usage:
    python scripts/grpo_train_hotel.py [MODEL]
"""

import os, sys, math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
from peft import get_peft_model, LoraConfig

MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-8B"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "../data/grpo_hotel_completions")
OUTPUT_DIR = os.path.join(BASE_DIR, "../outputs/grpo_hotel")

for i, arg in enumerate(sys.argv):
    if arg == "--dataset-path" and i + 1 < len(sys.argv):
        DATASET_PATH = sys.argv[i + 1]
    if arg == "--output-dir" and i + 1 < len(sys.argv):
        OUTPUT_DIR = sys.argv[i + 1]

# Hyperparams
EPOCHS = 3
BATCH_SIZE = 16
GRAD_ACCUM = 4
LR = 2e-5
WARMUP_RATIO = 0.1
MAX_SEQ_LEN = 512


def tokenize_example(example, tokenizer):
    """Tokenize prompt+completion, return input_ids and the completion mask."""
    messages = [
        {"role": "user", "content": example["prompt"]},
        {"role": "assistant", "content": example["completion"]},
    ]
    full = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    prompt_only = tokenizer.apply_chat_template(
        [{"role": "user", "content": example["prompt"]}],
        tokenize=False, add_generation_prompt=True,
    )

    full_ids = tokenizer.encode(full, add_special_tokens=False, max_length=MAX_SEQ_LEN, truncation=True)
    prompt_ids = tokenizer.encode(prompt_only, add_special_tokens=False, max_length=MAX_SEQ_LEN, truncation=True)

    # Completion starts after prompt
    completion_start = len(prompt_ids)

    return {
        "input_ids": full_ids,
        "completion_start": completion_start,
        "reward": example["reward"],
    }


def collate_fn(batch):
    """Pad and collate a batch."""
    max_len = max(len(b["input_ids"]) for b in batch)

    input_ids = []
    attention_mask = []
    completion_masks = []
    rewards = []

    for b in batch:
        ids = b["input_ids"]
        pad_len = max_len - len(ids)

        input_ids.append(ids + [0] * pad_len)
        attention_mask.append([1] * len(ids) + [0] * pad_len)

        # Mask: 1 for completion tokens, 0 for prompt/padding
        cmask = [0] * b["completion_start"] + [1] * (len(ids) - b["completion_start"]) + [0] * pad_len
        completion_masks.append(cmask)
        rewards.append(b["reward"])

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "completion_mask": torch.tensor(completion_masks, dtype=torch.float),
        "rewards": torch.tensor(rewards, dtype=torch.float),
    }


if __name__ == "__main__":
    print(f"Loading dataset from {DATASET_PATH}...", flush=True)
    dataset = load_from_disk(DATASET_PATH)
    print(f"Dataset size: {len(dataset)}", flush=True)

    print(f"Loading model {MODEL}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16)

    # Patch rotary embedding instances to compute on CPU (avoids ROCm HIP crash)
    import types
    patched = 0
    for name, module in model.named_modules():
        if "rotary" in name.lower() or type(module).__name__.endswith("RotaryEmbedding"):
            @torch.no_grad()
            def _safe_forward(self, x, position_ids):
                inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
                position_ids_expanded = position_ids[:, None, :].float()
                device = x.device
                freqs = (inv_freq_expanded.cpu() @ position_ids_expanded.cpu()).transpose(1, 2).to(device)
                emb = torch.cat((freqs, freqs), dim=-1)
                cos = emb.cos() * self.attention_scaling
                sin = emb.sin() * self.attention_scaling
                return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
            module.forward = types.MethodType(_safe_forward, module)
            patched += 1
    print(f"Patched {patched} rotary embedding instances for ROCm", flush=True)

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    # Tokenize dataset
    print("Tokenizing...", flush=True)
    tokenized = [tokenize_example(dataset[i], tokenizer) for i in range(len(dataset))]
    print(f"Tokenized {len(tokenized)} examples", flush=True)

    dataloader = DataLoader(tokenized, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    total_steps = (len(dataloader) // GRAD_ACCUM) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    global_step = 0

    # Normalize rewards per prompt group (GRPO-style baseline subtraction)
    # Group by prompt, subtract mean reward within group
    from collections import defaultdict
    prompt_rewards = defaultdict(list)
    for t in tokenized:
        # Use first 50 chars of input_ids as key (same prompt)
        key = tuple(t["input_ids"][:50])
        prompt_rewards[key].append(t)

    for key, group in prompt_rewards.items():
        rewards = [g["reward"] for g in group]
        mean_r = sum(rewards) / len(rewards)
        std_r = max((sum((r - mean_r)**2 for r in rewards) / len(rewards)) ** 0.5, 1e-8)
        for g in group:
            g["advantage"] = (g["reward"] - mean_r) / std_r

    print(f"Total steps: {total_steps}, warmup: {warmup_steps}", flush=True)
    print("Starting training...", flush=True)

    for epoch in range(EPOCHS):
        epoch_loss = 0
        epoch_batches = 0
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            completion_mask = batch["completion_mask"].to(device)
            rewards = batch["rewards"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            # Shift for next-token prediction
            shift_logits = logits[:, :-1, :]
            shift_labels = input_ids[:, 1:]
            shift_mask = completion_mask[:, 1:]

            # Per-token cross-entropy
            log_probs = F.log_softmax(shift_logits, dim=-1)
            token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)

            # Masked mean log prob per example
            masked_log_probs = (token_log_probs * shift_mask).sum(dim=1) / shift_mask.sum(dim=1).clamp(min=1)

            # REINFORCE: loss = -reward * log_prob
            loss = -(rewards * masked_log_probs).mean()
            loss = loss / GRAD_ACCUM
            loss.backward()

            epoch_loss += loss.item() * GRAD_ACCUM
            epoch_batches += 1

            if (batch_idx + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % 10 == 0:
                    avg_loss = epoch_loss / epoch_batches
                    print(f"  step {global_step}/{total_steps} loss={avg_loss:.4f} "
                          f"lr={scheduler.get_last_lr()[0]:.2e}", flush=True)

        avg_loss = epoch_loss / max(epoch_batches, 1)
        print(f"Epoch {epoch+1}/{EPOCHS} done, avg_loss={avg_loss:.4f}", flush=True)

        # Save checkpoint
        ckpt_dir = os.path.join(OUTPUT_DIR, f"checkpoint-epoch{epoch+1}")
        model.save_pretrained(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)

    # Save final
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Saved LoRA adapter to {OUTPUT_DIR}", flush=True)
    print("Training complete.", flush=True)
