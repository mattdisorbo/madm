"""Minimal SFT smoke test using a HuggingFace dataset."""
import torch
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"ROCm (HIP) available: {torch.version.hip is not None}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

model_name = "Qwen/Qwen3-0.6B"
dataset = load_dataset("trl-lib/Capybara", split="train[:100]")

trainer = SFTTrainer(
    model=model_name,
    train_dataset=dataset,
    args=SFTConfig(
        output_dir="outputs/smoke-test",
        max_steps=10,
        logging_steps=1,
    ),
)
trainer.train()
print("Smoke test passed!")
