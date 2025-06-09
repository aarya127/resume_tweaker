from sentence_transformers import SentenceTransformer, util
import numpy as np

model = SentenceTransformer("BAAI/bge-base-en-v1.5")

# Your resume points
resume_points = [
    "Developed dashboards in Tableau to visualize sales trends.",
    "Built a predictive model to forecast customer churn using Python.",
    "Collaborated with cross-functional teams to improve data quality.",
]

# Job description (paste or load full text)
job_description = """
We are seeking a data analyst proficient in Python and experience with BI tools like Tableau or Power BI.
The candidate should be able to forecast business trends and collaborate across departments to deliver insights.
"""

# Step 1: Encode all texts
resume_embeddings = model.encode(resume_points, convert_to_tensor=True)
job_embedding = model.encode(job_description, convert_to_tensor=True)

# Step 2: Compute cosine similarity for each point
similarity_scores = util.cos_sim(resume_embeddings, job_embedding)

# Step 3: Show results
for point, score in zip(resume_points, similarity_scores):
    print(f"Point: {point}\nSimilarity: {score.item():.3f}\n")
