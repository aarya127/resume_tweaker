import subprocess
import sys
import os

# Run the job description popup and save the summary to file
subprocess.run([sys.executable, os.path.join("job", "description.py")])

# After description.py runs, similarity.py will use the summary output in job_description.txt
subprocess.run([sys.executable, os.path.join("tools", "similarity.py")])

# You can add more script calls here if needed
