# Define the path to the LaTeX file
latex_file_path = "/Users/aaryas127/Documents/GitHub/resume_tweaker/ds.tex"

# Read the contents of the LaTeX file
try:
    with open(latex_file_path, "r") as file:
        latex_content = file.read()
        print("LaTeX file content successfully read!")
        print(latex_content)  # Print the content (optional)
except FileNotFoundError:
    print(f"Error: The file '{latex_file_path}' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")