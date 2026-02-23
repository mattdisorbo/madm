import threading
from transformers import pipeline

QWEN_MODEL_LARGE = "Qwen/Qwen2.5-7B-Instruct"
DEEPSEEK_MODEL   = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
GLM_MODEL        = "THUDM/glm-4-9b-chat"

PROMPT = "What is 2 + 2? Reply with a single number."

local_pipes = {}
local_locks = {}

for m in [QWEN_MODEL_LARGE, DEEPSEEK_MODEL, GLM_MODEL]:
    print(f"Loading {m}...", flush=True)
    local_pipes[m] = pipeline("text-generation", model=m, torch_dtype="auto", device_map="auto")
    local_locks[m] = threading.Lock()
    print(f"{m} loaded.", flush=True)

for m in [QWEN_MODEL_LARGE, DEEPSEEK_MODEL, GLM_MODEL]:
    with local_locks[m]:
        out = local_pipes[m]([{"role": "user", "content": PROMPT}], max_new_tokens=64)
    response = out[0]["generated_text"][-1]["content"]
    print(f"\n[{m}]\n{response}", flush=True)

print("\nSmoke test complete.", flush=True)
