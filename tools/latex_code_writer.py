import os
import sys
import json
from openai import OpenAI
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../model')))
from identifier import ResumeIdentifier
from approve_change import get_kept_resume_points

# Get the correct resume .txt file using ResumeIdentifier
identifier = ResumeIdentifier()
txt_file = identifier.get_txt_file()
resume_path = os.path.join(os.path.dirname(__file__), f'../resume/{txt_file}')

# Read the LaTeX resume code
with open(resume_path, 'r') as f:
    resume_latex = f.read()

# Get the kept resume points (approved changes)
kept_results = get_kept_resume_points()

# Compose the prompt for the IBM model
prompt = (
    "You are an expert LaTeX resume editor. Here is my resume in LaTeX format. "
    "Please update the resume by making the following changes: for each pair, replace the discarded point with the kept point. "
    "Here are the changes to make (kept/discarded pairs):\n" +
    '\n'.join([
        f"- Kept: {r['kept']}\n  Discarded: {r['discarded']}" for r in kept_results if r['discarded']
    ]) +
    "\n\nHere is my current LaTeX resume code:\n" + resume_latex
)

# Print the prompt for debugging/inspection
print("\n--- PROMPT TO DEEPSEEK MODEL ---\n")
print(prompt)
print("\n--- END PROMPT ---\n")

# Load API key (second line in keys.txt)
with open(os.path.join(os.path.dirname(__file__), '../keys.txt'), 'r') as f:
    lines = [line.strip() for line in f if line.strip()]
api_key = lines[1] if len(lines) > 1 else None

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=api_key
)

# Call the DeepSeek model (use the same model as in resume_breakdown.py)
completion = client.chat.completions.create(
    model="deepseek-ai/deepseek-r1-0528",
    messages=[{"role": "user", "content": prompt}],
    temperature=0.2,
    top_p=0.7,
    max_tokens=7000,
    stream=True
)

output_text = ""
print("\n--- MODEL OUTPUT (streaming) ---\n")
for chunk in completion:
    if hasattr(chunk, 'choices') and chunk.choices[0].delta.content is not None:
        output_text += chunk.choices[0].delta.content
        print(chunk.choices[0].delta.content, end='', flush=True)
    elif isinstance(chunk, dict) and 'choices' in chunk and chunk['choices'][0]['delta']['content'] is not None:
        output_text += chunk['choices'][0]['delta']['content']
        print(chunk['choices'][0]['delta']['content'], end='', flush=True)
    elif isinstance(chunk, str):
        output_text += chunk
        print(chunk, end='', flush=True)
print("\n--- END MODEL OUTPUT ---\n")

# Print only the revised LaTeX resume (strip leading/trailing whitespace)
latex_start = output_text.find('\\documentclass')
if latex_start != -1:
    latex_code = output_text[latex_start:]
else:
    latex_code = output_text
print("\n--- FINAL LATEX CODE ---\n")
print(latex_code.strip())
print("\n--- END FINAL LATEX CODE ---\n")

# Save only the LaTeX code to clipboard (macOS only)
try:
    import subprocess
    process = subprocess.Popen('pbcopy', env={'LANG': 'en_US.UTF-8'}, stdin=subprocess.PIPE)
    process.communicate(latex_code.strip().encode('utf-8'))
    print("\n[INFO] LaTeX code copied to clipboard.")
except Exception as e:
    print(f"[WARNING] Could not copy LaTeX code to clipboard: {e}")
