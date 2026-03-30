"""Download Qwen2.5-7B-Instruct to HF cache."""
import os
import json
from transformers import AutoModelForCausalLM, AutoTokenizer

ADAPTER_DIR = "data/together_hotel/adapter"
config = json.load(open(os.path.join(ADAPTER_DIR, "adapter_config.json")))
base_model = config["base_model_name_or_path"]

print(f"Downloading {base_model}...")
print("Tokenizer...")
AutoTokenizer.from_pretrained(base_model)
print("Model...")
AutoModelForCausalLM.from_pretrained(base_model)
print("Done!")
