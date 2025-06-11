# need atleast 3 occurances if a technology listed in the job description

import os
import csv
import json
from openai import OpenAI
from sentence_transformers import SentenceTransformer, util

# Load API key (second line in keys.txt, as in description.py)
with open(os.path.join(os.path.dirname(__file__), '../keys.txt'), 'r') as f:
    lines = [line.strip() for line in f if line.strip()]
api_key = lines[1] if len(lines) > 1 else None

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=api_key
)

# Load resume points from similarity_scores.csv (sorted least to greatest similarity)
sim_scores_path = os.path.join(os.path.dirname(__file__), '../resume/similarity_scores.csv')
resume_points = []
with open(sim_scores_path, 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        resume_points.append(row['Resume Point'])

# Load missing technologies from missing_technologies.txt
missing_tech_path = os.path.join(os.path.dirname(__file__), '../resume/missing_technologies.txt')
with open(missing_tech_path, 'r') as f:
    missing_techs = f.read()

# Compose the prompt for the model
prompt = (
    "You are an expert resume editor. Go through the following resume points (sorted by least similarity to the job description) and the list of missing technologies. "
    "For each missing technology, try to add it to the resume points, starting from the top. Ensure that each missing technology is mentioned once. "
    "You may replace existing technologies that are present in the resume but not in the job description. "
    "If a technology is present in my resume and the description, make sure it is mentioned 3 times atleast."
    "Once all missing technologies are addressed, stop editing. Do not go through all the resume points if not needed.\n\n"
    f"Missing technologies (with counts):\n{missing_techs}\n\n"
    f"Resume points (one per line):\n" + "\n".join(resume_points)
)

completion = client.chat.completions.create(
    model="deepseek-ai/deepseek-r1",
    messages=[{"role": "user", "content": prompt}],
    temperature=0.7,
    top_p=0.7,
    max_tokens=4096, 
    stream=True
)

output_text = ""
for chunk in completion:
    if chunk.choices[0].delta.content is not None:
        output_text += chunk.choices[0].delta.content
        print(chunk.choices[0].delta.content, end="")

# Optionally, save the revised resume points to a file
revised_path = os.path.join(os.path.dirname(__file__), '../resume/revised_resume_points.txt')
with open(revised_path, 'w') as f:
    f.write(output_text)

# After saving the revised resume points, compare similarity scores
model = SentenceTransformer("BAAI/bge-base-en-v1.5")

# Read the revised resume points
with open(revised_path, 'r') as f:
    revised_points = [line.strip() for line in f if line.strip()]

# Only compare for the same number of points as in the original
num_points = min(len(resume_points), len(revised_points))

# Read job summary from results.json (same as in similarity.py)
results_path = os.path.join(os.path.dirname(__file__), '../model/results.json')
with open(results_path, 'r') as f:
    results = json.load(f)
job_summary = results.get("primary_technical_skills", "") + "\n" + results.get("company_research", "")

# Compute similarity for original and revised points
original_scores = model.encode(resume_points[:num_points], convert_to_tensor=True)
revised_scores = model.encode(revised_points[:num_points], convert_to_tensor=True)
job_embedding = model.encode(job_summary, convert_to_tensor=True)

orig_sim = util.cos_sim(original_scores, job_embedding)
revised_sim = util.cos_sim(revised_scores, job_embedding)

# Write comparison to a CSV
compare_path = os.path.join(os.path.dirname(__file__), '../resume/revised_similarity_comparison.csv')
import csv
with open(compare_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Original Resume Point', 'Original Similarity', 'Revised Resume Point', 'Revised Similarity'])
    for i in range(num_points):
        writer.writerow([
            resume_points[i], f"{orig_sim[i].item():.3f}",
            revised_points[i], f"{revised_sim[i].item():.3f}"
        ])

# Use a second model call to explicitly list only the revised resume points
explicit_prompt = (
    "Given the following output from a resume editing model, explicitly list only the revised resume points, one per line, with no extra explanation or commentary.\n\n"
    f"Model output:\n{output_text}"
)

explicit_completion = client.chat.completions.create(
    model="nvidia/llama-3.1-nemotron-nano-vl-8b-v1",  # Use Nemotron model for explicit extraction
    messages=[{"role": "user", "content": explicit_prompt}],
    temperature=0.3,
    top_p=0.7,
    max_tokens=4096,
    stream=True
)

explicit_revised = ""
for chunk in explicit_completion:
    if chunk.choices[0].delta.content is not None:
        explicit_revised += chunk.choices[0].delta.content

# Save the explicit revised resume points to a file
explicit_revised_path = os.path.join(os.path.dirname(__file__), '../resume/explicit_revised_resume_points.txt')
with open(explicit_revised_path, 'w') as f:
    f.write(explicit_revised)

# Now do the similarity comparison using the explicit revised points
explicit_revised_points = [line.strip() for line in explicit_revised.split('\n') if line.strip()]
num_points = min(len(resume_points), len(explicit_revised_points))

original_scores = model.encode(resume_points[:num_points], convert_to_tensor=True)
revised_scores = model.encode(explicit_revised_points[:num_points], convert_to_tensor=True)
job_embedding = model.encode(job_summary, convert_to_tensor=True)

orig_sim = util.cos_sim(original_scores, job_embedding)
revised_sim = util.cos_sim(revised_scores, job_embedding)

compare_path = os.path.join(os.path.dirname(__file__), '../resume/explicit_revised_similarity_comparison.csv')
with open(compare_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Original Resume Point', 'Original Similarity', 'Explicit Revised Resume Point', 'Revised Similarity'])
    for i in range(num_points):
        writer.writerow([
            resume_points[i], f"{orig_sim[i].item():.3f}",
            explicit_revised_points[i], f"{revised_sim[i].item():.3f}"
        ])
