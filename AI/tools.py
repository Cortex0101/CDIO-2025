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

def create_txt_file_for_each_image(folder):
    """
    Creates an empty .txt file for each image in the specified folder. If one already exists, do not create a new one.
    """
    for filename in os.listdir(folder):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            txt_filename = os.path.splitext(filename)[0] + '.txt'
            txt_file_path = os.path.join(folder, txt_filename)
            if not os.path.exists(txt_file_path):
                with open(txt_file_path, 'w') as file:
                    file.write('')  # Create an empty .txt file
                    print(f"Created {txt_file_path} for {filename}") 
            else:
                print(f"File {txt_file_path} already exists, skipping creation.")
            

if __name__ == "__main__":
    # Example usage
    folder_path = 'AI/images'  # Replace with your folder path
    create_txt_file_for_each_image(folder_path)
    print("All .txt files in the folder have been cleared.")