import subprocess
import sys
import os

# Run the job description popup and save the summary to file
subprocess.run([sys.executable, os.path.join("model", "description.py")])

# Run the identifier to select the correct resume assets
subprocess.run([sys.executable, os.path.join("model", "identifier.py")])

# After resume_breakdown.py runs, similarity.py will use the summary output in results.json
subprocess.run([sys.executable, os.path.join("model", "similarity.py")])

# Run resume_breakdown to analyze the resume and job description
subprocess.run([sys.executable, os.path.join("model", "resume_breakdown.py")])

# Run tech_replacements to revise resume points and compare similarity
subprocess.run([sys.executable, os.path.join("tools", "tech_replacements.py")])

# Run approve_change to select which points to keep
subprocess.run([sys.executable, os.path.join("tools", "approve_change.py")])

# Run latex_code_writer to update the LaTeX resume
subprocess.run([sys.executable, os.path.join("tools", "latex_code_writer.py")])

# You can add more script calls here if needed
