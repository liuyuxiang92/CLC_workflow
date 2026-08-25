import os

def find_files_by_name(root_dir, filename):
    """
    Recursively find all files with the given name under the root directory.

    Parameters:
    - root_dir (str): The directory to start searching from
    - filename (str): The file name to look for (e.g., 'POSCAR')

    Returns:
    - List of full paths to the matched files
    """
    matched_paths = []
    for dirpath, _, files in os.walk(root_dir):
        if filename in files:
            full_path = os.path.join(dirpath, filename)
            matched_paths.append(full_path)
    return matched_paths

