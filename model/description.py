from openai import OpenAI
import tkinter as tk
from tkinter import simpledialog
import os

def get_job_description():
    # Create a pop-up window
    root = tk.Tk()
    root.withdraw()  # Hide the main tkinter window

    # Prompt the user for input in a pop-up dialog
    job_description = simpledialog.askstring("Job Description", "Please paste the job description below:")
    
    # Validate the input (no minimum length)
    if not job_description:
        tk.messagebox.showerror("Error", "Job description cannot be empty.")
        return get_job_description()  # Prompt again if invalid
    
    return job_description

# Example usage
job_description = get_job_description()

# Read the second API key from keys.txt (for NVIDIA/OpenAI)
with open(os.path.join(os.path.dirname(__file__), '../keys.txt'), 'r') as f:
    lines = [line.strip() for line in f if line.strip()]
api_key = lines[1] if len(lines) > 1 else None

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=api_key
)

results = {}

# First output: List primary technical skills
completion_technical_skills = client.chat.completions.create(
    model="nvidia/llama-3.1-nemotron-nano-vl-8b-v1",
    messages=[{"role": "user", "content": "list the primary technical skills necessary for this job description:" + job_description}],
    temperature=1.00,
    top_p=0.01,
    max_tokens=1024,
    stream=True
)

skills_output = ""
for chunk in completion_technical_skills:
    if chunk.choices[0].delta.content is not None:
        skills_output += chunk.choices[0].delta.content
results["primary_technical_skills"] = skills_output

# Second output: Research on the company
completion_company_research = client.chat.completions.create(
    model="nvidia/llama-3.1-nemotron-nano-vl-8b-v1",
    messages=[{"role": "user", "content": "Using the following job description, perform research on the company and provide insights about the technologies they use based on the position: " + job_description}],
    temperature=1.00,
    top_p=0.01,
    max_tokens=1024,
    stream=True
)

company_research_output = ""
for chunk in completion_company_research:
    if chunk.choices[0].delta.content is not None:
        company_research_output += chunk.choices[0].delta.content
results["company_research"] = company_research_output

# Optionally print or use the results dictionary
print("\nPrimary Technical Skills:")
print(results["primary_technical_skills"])
print("\n\nCompany Research:")
print(results["company_research"])
