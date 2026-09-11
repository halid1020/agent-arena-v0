import os
import glob

def combine_python_files(directory_path, output_filename="combined_code.txt"):
    """
    Finds all .py files in a directory and combines them into a single text file.
    """
    # Create a search pattern to find all .py files in the specified directory
    search_pattern = os.path.join(directory_path, "*.py")
    py_files = glob.glob(search_pattern)

    # Check if any files were found
    if not py_files:
        print(f"No Python files found in '{directory_path}'.")
        return

    try:
        # Open the output file in write mode
        with open(output_filename, 'w', encoding='utf-8') as outfile:
            for file_path in py_files:
                # Extract just the filename from the full path
                filename = os.path.basename(file_path)
                
                # Write a clear visual header for each file
                outfile.write(f"{'='*50}\n")
                outfile.write(f"FILE: {filename}\n")
                outfile.write(f"{'='*50}\n\n")
                
                # Read the contents of the current python file and write it
                try:
                    with open(file_path, 'r', encoding='utf-8') as infile:
                        outfile.write(infile.read())
                except Exception as e:
                    outfile.write(f"# Error reading {filename}: {e}\n")
                
                # Add some blank space before the next file begins
                outfile.write("\n\n\n")
                
        print(f"Success! Combined {len(py_files)} files into '{output_filename}'.")
    
    except Exception as e:
        print(f"An error occurred while writing the output file: {e}")

# --- Execution ---
if __name__ == "__main__":
    # The directory you want to scan. 
    # "." means the current directory where this script is located.
    # You can change this to a specific path like "C:/Users/Name/Documents/MyCode"
    target_directory = "." 
    
    # The name of the text file you want to generate
    output_file = "all_my_python_code.txt"
    
    combine_python_files(target_directory, output_file)