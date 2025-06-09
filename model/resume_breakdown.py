from openai import OpenAI
from identifier import ResumeIdentifier
import os

# Read the third API key from keys.txt (for resume_breakdown)
with open(os.path.join(os.path.dirname(__file__), '../keys.txt'), 'r') as f:
    lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]
api_key = lines[2] if len(lines) > 2 else None

identifier = ResumeIdentifier()
txt_file, pdf_file = identifier.get_assets()

# Read the LaTeX resume code
with open(os.path.join(os.path.dirname(__file__), f'../resume/{txt_file}'), 'r') as f:
    resume_latex = f.read()

client = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key = api_key
)

completion = client.chat.completions.create(
  model="deepseek-ai/deepseek-r1",
  messages=[{"role":"user","content":f"What are the key components of this resume? The user will provide a resume in latex. Please outline key technologies used, as well as core strengths.\n\nResume LaTeX code:\n{resume_latex}"}],
  temperature=0.6,
  top_p=0.7,
  max_tokens=4096,
  stream=True
)

for chunk in completion:
  if chunk.choices[0].delta.content is not None:
    print(chunk.choices[0].delta.content, end="")



