import os
from pathlib import Path


def dump_folder_contents(folder_path, output_filename, extensions):
    """
    Walks through a folder, finds files with specific extensions,
    and dumps their names and contents into a single file.
    """
    folder = Path(folder_path)

    # Normalize extensions to ensure they start with a dot (e.g., 'py' becomes '.py')
    extensions = tuple(ext if ext.startswith('.') else f'.{ext}' for ext in extensions)

    # Open the output file in write mode
    with open(output_filename, 'w', encoding='utf-8') as outfile:

        # Walk through the directory tree recursively
        for root, dirs, files in os.walk(folder):
            for file in files:
                # Check if the file matches our extension mask
                if file.endswith(extensions):
                    file_path = Path(root) / file

                    try:
                        # Read the source file
                        with open(file_path, 'r', encoding='utf-8') as infile:
                            content = infile.read()

                        # Write the formatting: file_name \n dump of file
                        outfile.write(f"{file_path}\n")
                        outfile.write(content)
                        outfile.write("\n\n")  # Added padding between files for readability

                    except UnicodeDecodeError:
                        print(f"Skipping {file_path}: Unable to decode as UTF-8 (likely a binary file).")
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")

    print(f"Successfully dumped contents to: {output_filename}")


# ==========================================
# Example Usage
# ==========================================
if __name__ == "__main__":
    # 1. Set the target folder path (use '.' for current directory)
    TARGET_FOLDER = Path("./core")

    # 2. Set the name of the final aggregated file
    OUTPUT_FILE = "core_dump.txt"

    # 3. Define the extensions you want to include
    ALLOWED_EXTENSIONS = ['.py', '.js', '.j2', '.md', '.txt']

    dump_folder_contents(TARGET_FOLDER, OUTPUT_FILE, ALLOWED_EXTENSIONS)