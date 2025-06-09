from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import disk_offload
import os

# Read all API keys from keys.txt (one per line)
with open(os.path.join(os.path.dirname(__file__), '../keys.txt'), 'r') as f:
    api_keys = [line.strip() for line in f if line.strip()]
# Example usage: use the first API key
api_key = api_keys[0] if api_keys else None

# Load the model with full weights
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B", device_map=None)

# Use disk offloading explicitly
disk_offload(model, offload_dir="/tmp/offload")  # Specify the directory for offloading

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")

prompt = "Explain how photosynthesis works."
inputs = tokenizer(prompt, return_tensors="pt").to("cpu")  # Ensure inputs are on CPU
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))