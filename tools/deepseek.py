from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-0528-Qwen3-8B")

# Load model in 4-bit quantized mode using bitsandbytes
model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
    device_map="auto",         # Automatically maps model to available GPU
    load_in_4bit=True,         # Enables 4-bit loading (uses bitsandbytes)
    torch_dtype="auto"
)

# Create text generation pipeline
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Define prompt
prompt = "Can you rewrite this resume summary to be more impactful: 'I worked at a bank doing data analysis.'"

# Generate output
output = pipe(prompt, max_new_tokens=150, do_sample=True, temperature=0.7)
print(output[0]["generated_text"])
