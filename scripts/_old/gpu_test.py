"""Minimal GPU forward pass test. Try different models/dtypes until one works."""
import os, sys, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-8B"
DTYPE = torch.float32 if "--fp32" in sys.argv else torch.bfloat16

print(f"Model: {MODEL}", flush=True)
print(f"Dtype: {DTYPE}", flush=True)
print(f"CUDA available: {torch.cuda.is_available()}", flush=True)
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

# Test 1: simple matmul on GPU
print("\n=== Test 1: Simple GPU matmul ===", flush=True)
a = torch.randn(4, 64, 1, dtype=torch.float32, device="cuda")
b = torch.randn(4, 1, 128, dtype=torch.float32, device="cuda")
c = a @ b
print(f"OK: matmul shape {c.shape}", flush=True)

# Test 2: bfloat16 matmul
print("\n=== Test 2: bf16 GPU matmul ===", flush=True)
a = torch.randn(4, 64, 1, dtype=torch.bfloat16, device="cuda")
b = torch.randn(4, 1, 128, dtype=torch.bfloat16, device="cuda")
c = a @ b
print(f"OK: bf16 matmul shape {c.shape}", flush=True)

# Test 3: Linear layer
print("\n=== Test 3: Linear layer ===", flush=True)
linear = torch.nn.Linear(128, 256, dtype=DTYPE).cuda()
x = torch.randn(2, 10, 128, dtype=DTYPE, device="cuda")
y = linear(x)
print(f"OK: linear shape {y.shape}", flush=True)

# Test 4: Load model and do forward pass
print(f"\n=== Test 4: Load {MODEL} ===", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=DTYPE)
model.to("cuda")
model.eval()
print("Model loaded on GPU", flush=True)

ids = tokenizer("Hello world", return_tensors="pt").to("cuda")
print(f"Input shape: {ids['input_ids'].shape}", flush=True)

with torch.no_grad():
    out = model(**ids)
print(f"OK: Forward pass! Output shape: {out.logits.shape}", flush=True)

# Test 5: backward pass
print("\n=== Test 5: Backward pass ===", flush=True)
model.train()
ids = tokenizer("Hello world", return_tensors="pt").to("cuda")
out = model(**ids)
loss = out.logits.mean()
loss.backward()
print(f"OK: Backward pass! Loss: {loss.item():.4f}", flush=True)

print("\n=== ALL TESTS PASSED ===", flush=True)
