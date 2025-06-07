from openai import OpenAI
import tkinter as tk
from tkinter import simpledialog

def get_job_description():
    # Create a pop-up window
    root = tk.Tk()
    root.withdraw()  # Hide the main tkinter window

    # Prompt the user for input in a pop-up dialog
    job_description = simpledialog.askstring("Job Description", "Please paste the job description below:")
    
    # Validate the input
    if not job_description or len(job_description) < 1000:
        tk.messagebox.showerror("Error", "Job description must be at least 1000 characters long.")
        return get_job_description()  # Prompt again if invalid
    
    return job_description

# Example usage
job_description = get_job_description()

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key="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"
)

# First output: List primary technical skills
completion_technical_skills = client.chat.completions.create(
    model="nvidia/llama-3.1-nemotron-nano-vl-8b-v1",
    messages=[{"role": "user", "content": "list the primary technical skills necessary for this job description:" + job_description}],
    temperature=1.00,
    top_p=0.01,
    max_tokens=1024,
    stream=True
)

print("\nPrimary Technical Skills:")
for chunk in completion_technical_skills:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")

# Second output: Research on the company
completion_company_research = client.chat.completions.create(
    model="nvidia/llama-3.1-nemotron-nano-vl-8b-v1",
    messages=[{"role": "user", "content": "Using the following job description, perform research on the company and provide insights about the technologies they use based on the position: " + job_description}],
    temperature=1.00,
    top_p=0.01,
    max_tokens=1024,
    stream=True
)

print("\n\nCompany Research:")
for chunk in completion_company_research:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")
