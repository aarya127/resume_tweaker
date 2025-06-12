from openai import OpenAI
from identifier import ResumeIdentifier
import os
import json
import re
from io import StringIO
import csv

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
    "Give me a short and simple list of all the technologies and the number of occurrences in my resume vs job description. "
    "Format the output as follows: "
    "**Overlap (Resume vs Job Description):**  \n"
    "- **Java** (1 vs 1)  \n"
    "- **C/C++** (2 vs 1)   etc, "
    "**Key Technologies Missing in Resume (Job Description):**  \n"
    "- **Ruby** (0 vs 1)  \n"
    "- **Swift** (0 vs 1) etc, "
    "**Technologies in Resume Not in Job Description:**  \n"
    "- **Perl** (1 vs 0)  \n"
    "- **COBOL** (1 vs 0) etc. "
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

# Ask Nemotron model for additional missing technologies based on job description and resume
nemotron_client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=api_key
)

nemotron_prompt = (
    "Using the job description and my resume, is there anything I will be working with that I don't fulfill in my resume? "
    f"Job description key items: {primary_technical_skills} and {company_research}\n"
    f"Resume LaTeX code: {resume_latex}"
)

nemotron_completion = nemotron_client.chat.completions.create(
    model="nvidia/llama-3.1-nemotron-nano-vl-8b-v1",
    messages=[{"role": "user", "content": nemotron_prompt}],
    temperature=0.3,
    top_p=0.7,
    max_tokens=1024,
    stream=True
)

nemotron_output = ""
for chunk in nemotron_completion:
    if chunk.choices[0].delta.content is not None:
        nemotron_output += chunk.choices[0].delta.content

# Save the Nemotron output to a TXT file (not CSV)
nemotron_txt_path = os.path.join(os.path.dirname(__file__), '../resume/nemotron_missing_technologies.txt')
with open(nemotron_txt_path, 'w') as txtfile:
    txtfile.write(nemotron_output.strip())



