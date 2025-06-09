from sentence_transformers import SentenceTransformer, util
import numpy as np
import os
import re
import pdfplumber
import sys
import tkinter as tk
from tkinter import simpledialog

model = SentenceTransformer("BAAI/bge-base-en-v1.5")

# Read PDF resume and extract bullet points
resume_points = []
pdf_path = os.path.join(os.path.dirname(__file__), '../resume/core_resume.pdf')
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

# Instead of reading the job description, read the summary output from description.py
summary_path = os.path.join(os.path.dirname(__file__), '../job_description.txt')
with open(summary_path, 'r') as f:
    job_summary = f.read().strip()

# Step 1: Encode all texts
resume_embeddings = model.encode(resume_points, convert_to_tensor=True)
job_embedding = model.encode(job_summary, convert_to_tensor=True)

# Step 2: Compute cosine similarity for each point
similarity_scores = util.cos_sim(resume_embeddings, job_embedding)

# Step 3: Show results
for point, score in zip(resume_points, similarity_scores):
    print(f"Point: {point}\nSimilarity: {score.item():.3f}\n")
