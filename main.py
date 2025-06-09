import subprocess
import sys
import os

# Run the job description popup and save the summary to file
subprocess.run([sys.executable, os.path.join("model", "description.py")])

# Run the identifier to select the correct resume assets
subprocess.run([sys.executable, os.path.join("model", "identifier.py")])

# After identifier.py runs, similarity.py will use the summary output in results.json
subprocess.run([sys.executable, os.path.join("model", "similarity.py")])

# You can add more script calls here if needed
