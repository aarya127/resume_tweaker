from openai import OpenAI
from identifier import ResumeIdentifier
import os
import json
import re
from io import StringIO

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

# Compose the prompt using results from results.json
results_path = os.path.join(os.path.dirname(__file__), 'results.json')
with open(results_path, 'r') as f:
    results = json.load(f)

primary_technical_skills = results.get("primary_technical_skills", "")
company_research = results.get("company_research", "")

missing_tech_prompt = (
    "Tell me the key technologies that are missing in my resume and are present in the job description or company's frequently used technologies and vice versa. "
    "Give me a short and simple list of all the technologies and the number of occurrences in my resume vs job description.\n\n"
    f"Job description key skills: {primary_technical_skills}\n"
    f"Resume LaTeX code: {resume_latex}"
)

completion = client.chat.completions.create(
  model="deepseek-ai/deepseek-r1",
  messages=[{"role":"user","content":missing_tech_prompt}],
  temperature=0.6,
  top_p=0.7,
  max_tokens=4096,
  stream=True
)

output_text = ""
for chunk in completion:
    if chunk.choices[0].delta.content is not None:
        output_text += chunk.choices[0].delta.content
        print(chunk.choices[0].delta.content, end="")

# After printing the model output, save the short list to a text file
text_path = os.path.join(os.path.dirname(__file__), '../resume/missing_technologies.txt')
with open(text_path, 'w') as txtfile:
    txtfile.write(output_text)



