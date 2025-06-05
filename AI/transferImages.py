import os

image_and_label_dir = "AI/images"

def transfer_images(folder_name, distribution=0.3):
    """
    Transfers images from the specified folder to the image_and_label_dir.

    distribution is the percentage of images to be transferred to the validation set.
    The rest will remain in the training set.
    """
    
    # validate that folder has a train and val folder
    train_folder = os.path.join(folder_name, "train")
    val_folder = os.path.join(folder_name, "val")
    if not os.path.exists(train_folder) or not os.path.exists(val_folder):
        raise ValueError(f"Folder {folder_name} must contain 'train' and 'val' subfolders.")
    
    # validate that train and val folders have images and labels folder
    train_images_folder = os.path.join(train_folder, "images")
    train_labels_folder = os.path.join(train_folder, "labels")

    val_images_folder = os.path.join(val_folder, "images")
    val_labels_folder = os.path.join(val_folder, "labels")

    if not os.path.exists(train_images_folder) or not os.path.exists(train_labels_folder):
        raise ValueError(f"Train folder {train_folder} must contain 'images' and 'labels' subfolders.")
    if not os.path.exists(val_images_folder) or not os.path.exists(val_labels_folder):
        raise ValueError(f"Val folder {val_folder} must contain 'images' and 'labels' subfolders.")
    
    # transfer 70% randomly selected images from image_and_label_dir to train_images_folder
    # select 70% of the .jpg files in image_and_label_dir
    all_images = [f for f in os.listdir(image_and_label_dir) if f.endswith('.jpg')]
    num_train_images = int(len(all_images) * (1 - distribution))
    train_images = all_images[:num_train_images]
    val_images = all_images[num_train_images:]

    # transfer images to train_images_folder
    for image in train_images:
        src_image_path = os.path.join(image_and_label_dir, image)
        dest_image_path = os.path.join(train_images_folder, image)
        if os.path.exists(src_image_path):
            os.rename(src_image_path, dest_image_path)

    # transfer images to val_images_folder
    for image in val_images:
        src_image_path = os.path.join(image_and_label_dir, image)
        dest_image_path = os.path.join(val_images_folder, image)
        if os.path.exists(src_image_path):
            os.rename(src_image_path, dest_image_path)

    # transfer labels to train_labels_folder and val_labels_folder, the have the same name but .txt extension
    for image in train_images:
        label_name = os.path.splitext(image)[0] + '.txt'
        src_label_path = os.path.join(image_and_label_dir, label_name)
        dest_label_path = os.path.join(train_labels_folder, label_name)
        if os.path.exists(src_label_path):
            os.rename(src_label_path, dest_label_path)

    for image in val_images:
        label_name = os.path.splitext(image)[0] + '.txt'
        src_label_path = os.path.join(image_and_label_dir, label_name)
        dest_label_path = os.path.join(val_labels_folder, label_name)
        if os.path.exists(src_label_path):
            os.rename(src_label_path, dest_label_path)

def cleanup(folder_name):
    """
    Cleans up the train and val folders by removing empty directories.
    """
    train_folder = os.path.join(folder_name, "train")
    val_folder = os.path.join(folder_name, "val")

    # remove all directories in train and val folders and recrate images and labels folders
    for folder in [train_folder, val_folder]:
        if os.path.exists(folder):
            for subfolder in os.listdir(folder):
                subfolder_path = os.path.join(folder, subfolder)
                # remove all files from subfolder
                if os.path.isdir(subfolder_path):
                    for file in os.listdir(subfolder_path):
                        file_path = os.path.join(subfolder_path, file)
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                    # remove the subfolder itself
                    os.rmdir(subfolder_path)
            # recreate images and labels folders
            os.makedirs(os.path.join(folder, "images"), exist_ok=True)
            os.makedirs(os.path.join(folder, "labels"), exist_ok=True)

def convert_labels_to_positive(folder):
    """
    goes through each .txt file in the folder

    contents could be

    0 0.216406 0.712500 0.020313 0.021875
1 0.214062 0.742188 0.018750 0.025000
1 0.365625 0.676562 0.025000 0.025000
1 0.320312 0.196094 0.028125 0.023438
1 0.553125 0.188281 0.021875 0.029687
1 0.708594 0.351562 0.023438 0.025000
1 0.745313 0.246094 0.031250 0.026562
1 0.771875 0.176563 0.018750 0.021875
1 0.775781 0.128906 0.020313 0.023438
2 0.628906 0.496875 0.032813 0.056250
3 0.506250 0.357812 0.090625 0.109375
4 0.312500 0.235937 0.103125 0.118750
5 0.841406 0.459375 0.026562 0.053125
6 0.153125 0.412500 0.031250 -0.103125
7 0.146094 0.418750 0.082812 -0.750000
7 0.856250 0.468750 0.040625 0.771875
7 0.510938 0.091406 0.756250 0.089063
7 0.486719 0.789844 0.773438 0.104688

    and if it sees a negative value, it will convert it to a positive value
    """""

    for filename in os.listdir(folder):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder, filename)
            with open(file_path, 'r') as file:
                lines = file.readlines()
            
            with open(file_path, 'w') as file:
                for line in lines:
                    parts = line.split()
                    if len(parts) > 1:
                        # Convert negative values to positive
                        parts[1:] = [str(abs(float(part))) for part in parts[1:]]
                        file.write(' '.join(parts) + '\n')

if __name__ == "__main__":
    folder_name = "AI/datasets/V5"
    convert_labels_to_positive(folder_name + "/train/labels")
    convert_labels_to_positive(folder_name + "/val/labels")
    
    
