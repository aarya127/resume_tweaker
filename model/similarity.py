from sentence_transformers import SentenceTransformer, util
import numpy as np
import os
import re
import sys
import tkinter as tk
from tkinter import simpledialog
import json
from identifier import ResumeIdentifier

model = SentenceTransformer("BAAI/bge-base-en-v1.5")

# Use the ResumeIdentifier class to get the correct PDF file
identifier = ResumeIdentifier()
_, pdf_file = identifier.get_assets()
pdf_path = os.path.join(os.path.dirname(__file__), f'../resume/{pdf_file}')

# Read PDF resume and extract bullet points
resume_points = []
import pdfplumber
with pdfplumber.open(pdf_path) as pdf:
    for page in pdf.pages:
        text = page.extract_text()
        if text:
            # Split lines and look for bullet-like lines
            for line in text.split('\n'):
                line = line.strip()
                # Common bullet characters: •, -, *, or lines starting with a number/letter and a dot
                if line.startswith(('•', '-', '*')):
                    resume_points.append(line.lstrip('•-*').strip())
                # Optionally, add more heuristics for bullet points

# If no bullet points found, fallback to all non-empty lines
if not resume_points:
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                for line in text.split('\n'):
                    if line.strip():
                        resume_points.append(line.strip())

# Read primary_technical_skills and company_research from results.json
results_path = os.path.join(os.path.dirname(__file__), '../model/results.json')
with open(results_path, 'r') as f:
    results = json.load(f)

job_summary = results.get("primary_technical_skills", "") + "\n" + results.get("company_research", "")

# Step 1: Encode all texts
resume_embeddings = model.encode(resume_points, convert_to_tensor=True)
job_embedding = model.encode(job_summary, convert_to_tensor=True)

# Step 2: Compute cosine similarity for each point
similarity_scores = util.cos_sim(resume_embeddings, job_embedding)

# Step 3: Store results in a CSV sorted by similarity (least to greatest)
import csv

results_list = [(point, score.item()) for point, score in zip(resume_points, similarity_scores)]
results_list.sort(key=lambda x: x[1])  # Sort by similarity score ascending

csv_path = os.path.join(os.path.dirname(__file__), '../resume/similarity_scores.csv')
with open(csv_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Resume Point', 'Similarity Score'])
    for point, score in results_list:
        writer.writerow([point, f"{score:.3f}"])
