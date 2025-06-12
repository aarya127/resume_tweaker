import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../model')))
from identifier import ResumeIdentifier

# Use ResumeIdentifier to get the correct resume .txt file
identifier = ResumeIdentifier()
txt_file = identifier.get_txt_file()

# Print all the revised resume points for the selected resume
# If you want to use a file like revised_resume_points_<resume>.txt, adjust here
revised_path = os.path.join(os.path.dirname(__file__), f'../resume/revised_resume_points.txt')
with open(revised_path, 'r') as f:
    for line in f:
        if line.strip():
            print(line.strip())