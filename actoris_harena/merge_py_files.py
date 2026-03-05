import os

def merge_py_files(source_dir, output_file, ignore_folders=None):
    """
    Recursively reads .py files from source_dir and writes them to output_file.
    
    :param source_dir: The root directory to start searching.
    :param output_file: The path to the file where contents will be saved.
    :param ignore_folders: A list of folder names to exclude from the search.
    """
    if ignore_folders is None:
        ignore_folders = []

    # Convert ignore list to a set for faster lookups
    ignore_set = set(ignore_folders)

    # Open the output file in write mode
    with open(output_file, 'w', encoding='utf-8') as outfile:
        
        # Traverse the directory tree
        for root, dirs, files in os.walk(source_dir):
            
            # Modify dirs in-place to skip ignored directories.
            # This prevents os.walk from even entering the ignored folders.
            dirs[:] = [d for d in dirs if d not in ignore_set]

            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    
                    try:
                        with open(file_path, 'r', encoding='utf-8') as infile:
                            content = infile.read()
                            
                            # Write a clear header for each file so you know where the code came from
                            outfile.write(f"# {'='*60}\n")
                            outfile.write(f"# File: {file_path}\n")
                            outfile.write(f"# {'='*60}\n\n")
                            
                            # Write the actual file content
                            outfile.write(content)
                            outfile.write("\n\n")
                            
                    except Exception as e:
                        print(f"Skipped {file_path} due to error: {e}")
                        
    print(f"Successfully merged files into {output_file}")

# --- Example Usage ---
if __name__ == "__main__":
    # Directory to scan ('.' means current directory)
    TARGET_DIRECTORY = "." 
    
    # The file where everything will be pasted
    OUTPUT_FILENAME = "merged_code.txt" 
    
    # Folders to completely ignore (e.g., virtual environments, caches)
    FOLDERS_TO_IGNORE = [".venv", "venv", "__pycache__", ".git", "node_modules", "PyFlex", "real"]

    merge_py_files(
        source_dir=TARGET_DIRECTORY, 
        output_file=OUTPUT_FILENAME, 
        ignore_folders=FOLDERS_TO_IGNORE
    )