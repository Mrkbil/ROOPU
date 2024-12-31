import os

def rename_files_in_directory(directory):
    if not os.path.isdir(directory):
        print(f"Error: The directory '{directory}' does not exist.")
        return

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if not os.path.isfile(file_path):
            continue

        name, extension = os.path.splitext(filename)

        new_name = ''.join(filter(str.isdigit, name))[:4]

        if not new_name:
            print(f"Skipping '{filename}' (no digits found).")
            continue

        new_file_path = os.path.join(directory, new_name + extension)

        os.rename(file_path, new_file_path)
        print(f"Renamed: '{filename}' -> '{new_name + extension}'")

# Example usage
directory = "out2/"
rename_files_in_directory(directory)