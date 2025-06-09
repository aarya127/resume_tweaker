from openai import OpenAI
import os

# Read the third API key from keys.txt (for resume_breakdown)
with open(os.path.join(os.path.dirname(__file__), '../keys.txt'), 'r') as f:
    lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]
api_key = lines[2] if len(lines) > 2 else None

client = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key = api_key
)

completion = client.chat.completions.create(
  model="deepseek-ai/deepseek-r1",
  messages=[{"role":"user","content":"What are the key components of a resume? Please provide a detailed breakdown of each section, including the purpose and content typically included in each part."}],
  temperature=0.6,
  top_p=0.7,
  max_tokens=4096,
  stream=True
)

for chunk in completion:
  if chunk.choices[0].delta.content is not None:
    print(chunk.choices[0].delta.content, end="")

