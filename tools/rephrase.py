## NEED TO CHANGE THIS 


import os
import sys
import json
from approve_change import get_kept_resume_points
from openai import OpenAI

# Load job/role description from results.json
with open(os.path.join(os.path.dirname(__file__), '../model/results.json'), 'r') as f:
    job_desc = json.load(f)
description = job_desc.get('primary_technical_skills', '')

# Load DeepSeek API key
with open(os.path.join(os.path.dirname(__file__), '../keys.txt'), 'r') as f:
    lines = [line.strip() for line in f if line.strip()]
api_key = lines[1] if len(lines) > 1 else None

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=api_key
)

def rephrase_with_deepseek(point, description):
    prompt = (
        f"You are an expert resume writer. Rephrase the following resume bullet point so that it is between 98 and 106 characters, "
        f"while preserving its meaning and making it relevant to this job description: {description}\n\n"
        "Ensure to include the technologies present in the already revised point.\n\n"
        f"Bullet point: {point}\n\nRephrased bullet point (between 98 and 106 characters):"
    )
    response = client.chat.completions.create(
        model="deepseek-ai/deepseek-r1",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        top_p=0.7,
        max_tokens=2000,
        stream=False
    )
    # Extract the rephrased text
    content = response.choices[0].message.content.strip()
    # Only return the first line if multiple lines
    return content.split('\n')[0]

def get_similarity(a, b):
    set_a = set(a.lower().split())
    set_b = set(b.lower().split())
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)

kept_results = get_kept_resume_points()

updated_results = []
for result in kept_results:
    revised = result['kept']
    if revised is None or 98 <= len(revised) <= 106:
        updated_results.append(result)
        continue
    # Rephrase using DeepSeek only if not in the character count
    new_revised = rephrase_with_deepseek(revised, description)
    # Only update if similarity is higher
    old_sim = get_similarity(revised, description)
    new_sim = get_similarity(new_revised, description)
    if new_sim > old_sim:
        updated = result.copy()
        updated['kept'] = new_revised
        updated_results.append(updated)
    else:
        updated_results.append(result)

# Print updated results
for r in updated_results:
    print(f"KEPT: {r['kept']} | DISCARDED: {r['discarded']} | KEPT_SCORE: {r['kept_score']}")
