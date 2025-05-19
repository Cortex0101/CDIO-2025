import os
import glob
import cv2

# ─── CONFIG ────────────────────────────────────────────────────────────────
DATASET_DIR   = os.path.join("AI", "datasets")
IMG_DIR       = os.path.join(DATASET_DIR, "images")
LBL_DIR       = os.path.join(DATASET_DIR, "labels")
OUTPUT_FILE   = os.path.join(DATASET_DIR, "positives.txt")
IMG_EXT       = ".jpg"    # change if your images are .png, etc.
# ────────────────────────────────────────────────────────────────────────────

# 1) Get image size dynamically (in case they differ):
#    Here we assume at least one image exists.
sample_img = glob.glob(os.path.join(IMG_DIR, "*" + IMG_EXT))[0]
h_img, w_img = cv2.imread(sample_img).shape[:2]

with open(OUTPUT_FILE, "w") as out_f:
    # 2) Loop over every label file
    for lbl_path in glob.glob(os.path.join(LBL_DIR, "*.txt")):
        # 2a) Derive the image filename from the label filename
        base, _ = os.path.splitext(os.path.basename(lbl_path))
        img_name = base + IMG_EXT

        # 3) Read each polygon line
        with open(lbl_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                # parts[0] is class_id; parts[1:] are x1 y1 x2 y2 ... xN yN (normalized)
                coords = list(map(float, parts[1:]))

                # 4) De-normalize into pixel coords
                xs = [coords[i] * w_img for i in range(0, len(coords), 2)]
                ys = [coords[i] * h_img for i in range(1, len(coords), 2)]

                # 5) Get tight bounding box
                x_min, x_max = int(min(xs)), int(max(xs))
                y_min, y_max = int(min(ys)), int(max(ys))
                w_box, h_box = x_max - x_min, y_max - y_min

                # 6) Write one line per object in OpenCV format:
                #    <relative_image_path> <num_objects> <x> <y> <w> <h>
                #    (we write 1 for <num_objects> because each line is one ball)
                rel_img_path = os.path.join("images", img_name)
                out_f.write(f"{rel_img_path} 1 {x_min} {y_min} {w_box} {h_box}\n")

print(f"✔ Generated {OUTPUT_FILE} with all bounding boxes.")
