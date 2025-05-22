import os
import glob
import cv2

# ───────── CONFIG ─────────────────────────────────────────────────────────
version = "V3"
DATASET_DIR   = os.path.join("AI", "datasets", version)
TEST_DIR    = os.path.join(DATASET_DIR, "test")
TRAIN_DIR   = os.path.join(DATASET_DIR, "train")
VALID_DIR   = os.path.join(DATASET_DIR, "valid")
TRAIN_DIR  = os.path.join(DATASET_DIR, "train")
TEST_DIR = os.path.join(DATASET_DIR, "test")

IMG_FOLDER = "images"
LABEL_FOLDER = "labels"

# ──────────────────────────────────────────────────────────────────────────

# Go through all labels in test train and valid folders 
# and convert them to the new format
def convert_labels(folder):
    # Get all label files in the folder
    label_files = glob.glob(os.path.join(folder, LABEL_FOLDER, "*.txt"))

    for label_file in label_files:
        # Read the label file
        with open(label_file, "r") as f:
            lines = f.readlines()

        # Convert the labels to the new format
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue  # Skip invalid lines
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])

            if class_id == 0:
                # orange ball
                class_id = 0
            elif class_id == 4:
                # white ball
                class_id = 1
            else:
                continue

            # Convert to new format (YOLO format)
            new_line = f"{class_id} {x_center} {y_center} {width} {height}\n"
            new_lines.append(new_line)

        # Write the new labels to the file
        with open(label_file, "w") as f:
            f.writelines(new_lines)

# Convert labels in train, test and valid folders
convert_labels(os.path.join(TRAIN_DIR))	
convert_labels(os.path.join(TEST_DIR))
convert_labels(os.path.join(VALID_DIR))