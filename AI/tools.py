import os
import cv2
import numpy as np
from ultralytics import YOLO

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

colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (0, 0, 0), (255, 255, 255),
        (0, 128, 0), (128, 0, 128), (128, 128, 0), (128, 128, 128)
    ]

def visualize_model_on_image(img_path, model_path):
    """
    Visualizes the segmentation masks from a YOLO model on a single image.
    """
    # Load the model
    model = YOLO(model_path)

    # Read the image
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Could not read image {img_path}")
        return

    # Run inference
    results = model.predict(source=img, conf=0.3, iou=0.5)
    r = results[0]

    # Loop over masks
    if r.masks is not None:
        masks = r.masks.data.cpu().numpy()  # shape: (num_objects, H, W)

        for i, mask in enumerate(masks):
            cls = int(r.boxes.cls[i])
            conf = r.boxes.conf[i].item()
            color = colors[cls % len(colors)]
            color_bgr = (color[2], color[1], color[0])

            # Convert mask to binary and then to contours
            binary_mask = (mask > 0.5).astype(np.uint8) * 255
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Draw mask contour
            cv2.drawContours(img, contours, -1, color_bgr, 2)

            # Draw label at the top-left of the bounding box
            x1, y1, x2, y2 = map(int, r.boxes.xyxy[i])
            label = f"{model.names[cls]} {conf:.2f}"
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 2)

    else:
        print("No masks found in prediction.")

    # Show the image with masks and labels
    cv2.imshow('Segmented Detections', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def show_model_predictions_in_folder(folder_path, model_path):
    # Load YOLO model
    model = YOLO(model_path)

    # Get all .jpg files in the folder
    image_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith('.jpg')])
    if not image_files:
        print("No .jpg images found in the folder.")
        return

    # Define class colors
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (0, 0, 0), (255, 255, 255),
        (0, 128, 0), (128, 0, 128), (128, 128, 0), (128, 128, 128)
    ]

    index = 0
    while True:
        # Load image
        img_path = os.path.join(folder_path, image_files[index])
        img = cv2.imread(img_path)
        if img is None:
            print(f"Could not load image: {img_path}")
            continue

        # Predict with YOLO
        results = model.predict(source=img, conf=0.3, iou=0.5)
        r = results[0]

        # Draw segmentation masks and labels
        if r.masks is not None:
            masks = r.masks.data.cpu().numpy()
            for i, mask in enumerate(masks):
                cls = int(r.boxes.cls[i])
                conf = r.boxes.conf[i].item()

                color = colors[cls % len(colors)]
                color_bgr = (color[2], color[1], color[0])

                binary_mask = (mask > 0.5).astype(np.uint8) * 255
                contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(img, contours, -1, color_bgr, 2)

                x1, y1, x2, y2 = map(int, r.boxes.xyxy[i])
                label = f"{model.names[cls]} {conf:.2f}"
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 2)

        display_name = f"{index+1}/{len(image_files)}: {image_files[index]}"
        cv2.imshow(display_name, img)

        key = cv2.waitKey(0) & 0xFF
        cv2.destroyAllWindows()

        if key == ord('m'):  # next
            index = (index + 1) % len(image_files)
        elif key == ord('n'):  # previous
            index = (index - 1) % len(image_files)
        elif key == ord('q'):
            print("Exiting...")
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    show_model_predictions_in_folder('AI/images', 'ball_detect/v7dtu4/weights/best.pt')