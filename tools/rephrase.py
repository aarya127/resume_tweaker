# Use a pipeline to rephrase the resume points that have been changed. Check if makes sense. Also limit characters to 1 line
from transformers import pipeline
import torch

pipe = pipeline("text2text-generation", model="google/flan-t5-small", device=torch.device("cpu"))

# Prompt for rephrasing
prompt = "Rephrase the following resume point to emphasize C++: Designed 15+ Tableau & PowerBI Dashboards & wrote 20+ SQL queries to influence business decisions."

output = pipe(prompt, max_new_tokens=100, do_sample=True, temperature=0.7)
print(output[0]["generated_text"])