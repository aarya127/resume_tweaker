# Map a position string to the correct resume file name and PDF name
import os
import json

class ResumeIdentifier:
    def __init__(self, results_path=None):
        if results_path is None:
            results_path = os.path.join(os.path.dirname(__file__), 'results.json')
        self.results_path = results_path
        self._load_results()

    def _load_results(self):
        with open(self.results_path, 'r') as f:
            self.results = json.load(f)
        self.position_match = self.results.get("position_title_and_closest_match", "Data Science")

    def get_assets(self):
        if self.position_match in ["Data Science", "Machine Learning Engineering"]:
            return ["data_science.txt", "Aaryads.pdf"]
        elif self.position_match == "Data Engineer":
            return ["data_engineer.txt", "Aaryade.pdf"]
        elif self.position_match == "Data Science+Engineering":
            return ["mix.txt", "Aaryamix.pdf"]
        elif self.position_match == "Software Engineer":
            return ["swe.txt", "Aaryase.pdf"]
        else:
            return ["mix.txt", "Aaryamix.pdf"]  # default fallback

    def get_txt_file(self):
        return self.get_assets()[0]

    def get_pdf_file(self):
        return self.get_assets()[1]

# Example usage:
identifier = ResumeIdentifier()
txt_file, pdf_file = identifier.get_assets()
print(txt_file, pdf_file)
