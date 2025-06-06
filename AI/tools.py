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
            
dataset_folder = 'AI/datasets'  # Replace with your dataset folder path

def create_new_dataset_structure(name):
    """
    Creates a new dataset structure with 'train' and 'val' folders.
    """
    dir_path = "AI/datasets"

    # check if folder has folder with name
    if not os.path.exists(os.path.join(dir_path, name)):
        os.makedirs(os.path.join(dir_path, name))

    train_folder = os.path.join(dataset_folder, name, 'train')
    val_folder = os.path.join(dataset_folder, name, 'val')

    os.makedirs(os.path.join(train_folder, 'images'), exist_ok=True)
    os.makedirs(os.path.join(train_folder, 'labels'), exist_ok=True)
    os.makedirs(os.path.join(val_folder, 'images'), exist_ok=True)
    os.makedirs(os.path.join(val_folder, 'labels'), exist_ok=True)

    print(f"Created dataset structure for {name} in {dataset_folder}")

def copy_images_and_labels(source_folder, target_folder, distribution=0.3, first_n=0, last_n=250):
    """
    Copies images and labels from source_folder "image_0.jpg, image_1.jpg, ..., image_n.jpg"
    and image_0.txt, image_1.txt, ..., image_n.txt to target_folder.

    The images are split into training and validation sets based on the distribution ratio.

    Copy only image_first_n.jpg, image_first_n.txt, ..., image_last_n.jpg, image_last_n.txt
    """
    image_and_label_dir = source_folder
    train_images_folder = os.path.join(target_folder, 'train', 'images')
    train_labels_folder = os.path.join(target_folder, 'train', 'labels')
    val_images_folder = os.path.join(target_folder, 'val', 'images')
    val_labels_folder = os.path.join(target_folder, 'val', 'labels')

    if not os.path.exists(image_and_label_dir):
        print(f"Source folder {image_and_label_dir} does not exist.")
        return

    all_images = [f for f in os.listdir(image_and_label_dir) if f.endswith('.jpg')]
    
    # sort images by the _n in the filename
    all_images.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    all_images = all_images[first_n:last_n + 1]  # Select images from first_n to last_n

    num_train_images = int(len(all_images) * (1 - distribution))
    train_images = all_images[:num_train_images]
    val_images = all_images[num_train_images:]

    # Ensure target folders exist
    os.makedirs(train_images_folder, exist_ok=True)
    os.makedirs(train_labels_folder, exist_ok=True)
    os.makedirs(val_images_folder, exist_ok=True)
    os.makedirs(val_labels_folder, exist_ok=True)

    # Copy images to train_images_folder (dont delete existing files)
    for image in train_images:
        src_image_path = os.path.join(image_and_label_dir, image)
        dest_image_path = os.path.join(train_images_folder, image)
        if os.path.exists(src_image_path):
            if not os.path.exists(dest_image_path):
                os.rename(src_image_path, dest_image_path)
                print(f"Copied {src_image_path} to {dest_image_path}")
    
    # Copy images to val_images_folder (dont delete existing files)
    for image in val_images:
        src_image_path = os.path.join(image_and_label_dir, image)
        dest_image_path = os.path.join(val_images_folder, image)
        if os.path.exists(src_image_path):
            if not os.path.exists(dest_image_path):
                os.rename(src_image_path, dest_image_path)
                print(f"Copied {src_image_path} to {dest_image_path}")
    
    # Copy labels to train_labels_folder and val_labels_folder
    for image in train_images:
        label_name = os.path.splitext(image)[0] + '.txt'
        src_label_path = os.path.join(image_and_label_dir, label_name)
        dest_label_path = os.path.join(train_labels_folder, label_name)
        if os.path.exists(src_label_path):
            if not os.path.exists(dest_label_path):
                os.rename(src_label_path, dest_label_path)
                print(f"Copied {src_label_path} to {dest_label_path}")

    for image in val_images:
        label_name = os.path.splitext(image)[0] + '.txt'
        src_label_path = os.path.join(image_and_label_dir, label_name)
        dest_label_path = os.path.join(val_labels_folder, label_name)
        if os.path.exists(src_label_path):
            if not os.path.exists(dest_label_path):
                os.rename(src_label_path, dest_label_path)
                print(f"Copied {src_label_path} to {dest_label_path}")

def visualize_model_on_image(img_path, model):
    """
    Visualizes the segmentation model on a single image.
    """
    import cv2
    from ultralytics import YOLO

    # Load the model
    model = YOLO(model)

    # Read the image
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Could not read image {img_path}")
        return

    # Run inference
    results = model.predict(source=img, conf=0.3, iou=0.5)
    r = results[0]

    # Loop over each detected box
    for box in r.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        conf = box.conf[0].cpu().item()

        # Draw rectangle and confidence
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"{conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Display the image with detections
    cv2.imshow("Detections", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    copy_images_and_labels('AI/images', 'AI/datasets/V7', distribution=0.3, first_n=0, last_n=159)