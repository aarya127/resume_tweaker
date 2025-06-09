from openai import OpenAI
import os
import json

def get_job_description():
    import tkinter as tk
    from tkinter import simpledialog
    root = tk.Tk()
    root.withdraw()
    job_description = simpledialog.askstring("Job Description", "Please paste the job description below:")
    if not job_description:
        tk.messagebox.showerror("Error", "Job description cannot be empty.")
        return get_job_description()
    return job_description

if __name__ == "__main__":
    job_description = get_job_description()
    # Always overwrite job_description.txt with the new input
    with open("/Users/aaryas127/Documents/GitHub/resume_tweaker/job_description.txt", "w") as f:
        f.write(job_description)

# Read the second and fourth API keys from keys.txt
with open(os.path.join(os.path.dirname(__file__), '../keys.txt'), 'r') as f:
    lines = [line.strip() for line in f if line.strip()]
api_key_1 = lines[1] if len(lines) > 1 else None  # For first output
api_key_2 = lines[3] if len(lines) > 3 else None  # For second output

client_1 = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=api_key_1
)

client_2 = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=api_key_2
)

results = {}

# First output: List primary technical skills
completion_technical_skills = client_1.chat.completions.create(
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
completion_company_research = client_2.chat.completions.create(
    model="nvidia/llama-3.1-nemotron-nano-vl-8b-v1",
    messages=[{"role": "user", "content": "Using the following job title and company info, give me technologies they would use: " + job_description}],
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

# Third output: Position title and closest match
completion_position_title = client_1.chat.completions.create(
    model="nvidia/llama-3.1-nemotron-nano-vl-8b-v1",
    messages=[{"role": "user", "content": "which one of the following is the job title the closest to? Data Science, Data Engineering, Data Science+Engineering, Machine Learning Engineering or Software Engineer? Give me an answer from the options without an explanation. " + job_description}],
    temperature=1.00,
    top_p=0.01,
    max_tokens=256,
    stream=True
)

position_title_output = ""
for chunk in completion_position_title:
    if chunk.choices[0].delta.content is not None:
        position_title_output += chunk.choices[0].delta.content
results["position_title_and_closest_match"] = position_title_output

# Save results to JSON file (always overwrite)
with open(os.path.join(os.path.dirname(__file__), 'results.json'), 'w') as f:
    json.dump(results, f, indent=2)

# Optionally print or use the results dictionary
print("\nPrimary Technical Skills:")
print(results["primary_technical_skills"])
print("\n\nCompany Research:")
print(results["company_research"])
print("\nPosition Title and Closest Match:")
print(results["position_title_and_closest_match"])
