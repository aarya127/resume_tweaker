import os
import csv
from openai import OpenAI

# Paths to the CSV files
sim_scores_path = os.path.join(os.path.dirname(__file__), '../resume/similarity_scores.csv')
revised_sim_path = os.path.join(os.path.dirname(__file__), '../resume/revised_resume_points_similarity.csv')

# Load API key (second line in keys.txt, as in description.py)
with open(os.path.join(os.path.dirname(__file__), '../keys.txt'), 'r') as f:
    lines = [line.strip() for line in f if line.strip()]
api_key = lines[1] if len(lines) > 1 else None

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=api_key
)

# Read original resume points and scores
with open(sim_scores_path, 'r') as f:
    reader = csv.DictReader(f)
    orig_points = [row['Resume Point'].strip() for row in reader]
    f.seek(0)
    reader = csv.DictReader(f)
    orig_scores = [float(row['Similarity Score']) for row in reader]

# Read revised resume points and scores
with open(revised_sim_path, 'r') as f:
    reader = csv.DictReader(f)
    revised_points = [row['Revised Resume Point'].strip() for row in reader]
    f.seek(0)
    reader = csv.DictReader(f)
    revised_scores = [float(row['Similarity Score']) for row in reader]

# Use IBM Granite model to match revised points to original points and compare scores
for i, revised in enumerate(revised_points):
    prompt = (
        "Given the following list of original resume points, which one is most likely the source for the revised resume point below?\n"
        "List of original resume points:\n" + '\n'.join(f"{j+1}. {pt}" for j, pt in enumerate(orig_points)) +
        f"\n\nRevised resume point:\n{revised}\n\n"
        "Respond with only the number of the most likely original resume point."
        "Then give me a response in this format -- Original: , Revised:  , Original Score:  , Revised Score:  , Improved:  \n"
    )
    print("\n--- Resume Prompt Sent to Model ---\n")
    print(prompt)
    print("\n--- End Resume Prompt ---\n")
    completion = client.chat.completions.create(
        model="ibm/granite-3.3-8b-instruct",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        top_p=0.7,
        max_tokens=2048,
        stream=True
    )
    response = ""
    for chunk in completion:
        if hasattr(chunk, 'choices') and chunk.choices[0].delta.content is not None:
            response += chunk.choices[0].delta.content
        elif isinstance(chunk, dict) and 'choices' in chunk and chunk['choices'][0]['delta']['content'] is not None:
            response += chunk['choices'][0]['delta']['content']
        elif isinstance(chunk, str):
            response += chunk
    print(response)
