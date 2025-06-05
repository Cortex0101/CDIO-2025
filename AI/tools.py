import os

def clear_txt_files_content(folder):
    """
    Clears the content of all .txt files in the specified folder.
    """
    for filename in os.listdir(folder):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder, filename)
            with open(file_path, 'w') as file:
                file.write('')  # Clear the content of the file
            print(f"Cleared content of {file_path}")

if __name__ == "__main__":
    # Example usage
    folder_path = 'AI/images2'  # Replace with your folder path
    clear_txt_files_content(folder_path)
    print("All .txt files in the folder have been cleared.")