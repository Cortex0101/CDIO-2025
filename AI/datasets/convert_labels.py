import os
import glob
import cv2

# ───────── CONFIG ─────────────────────────────────────────────────────────
DATASET_DIR   = os.path.join("AI", "datasets")
IMG_DIR       = os.path.join(DATASET_DIR, "images")
SRC_LABEL_DIR = os.path.join(DATASET_DIR, "labels")       # your original polygon labels
DST_LABEL_DIR = os.path.join(DATASET_DIR, "labels_yolo")  # where converted labels go
IMG_EXT       = ".jpg"  # change if your images are .png, etc.
# ───────────────────────────────────────────────────────────────────────────

os.makedirs(DST_LABEL_DIR, exist_ok=True)

for lbl_path in glob.glob(os.path.join(SRC_LABEL_DIR, f"*{'.txt'}")):
    base = os.path.basename(lbl_path)                   # e.g. WIN_... .txt
    img_name = base.replace(".txt", IMG_EXT)            # same base → .jpg
    img_path = os.path.join(IMG_DIR, img_name)

    # load image to get width/height
    img = cv2.imread(img_path)
    if img is None:
        print(f"[WARN] no image for {lbl_path}, skipping")
        continue
    H, W = img.shape[:2]

    out_path = os.path.join(DST_LABEL_DIR, base)
    with open(lbl_path, "r") as fin, open(out_path, "w") as fout:
        for line in fin:
            parts = line.strip().split()
            orig_c = int(parts[0])

            # remap classes: 0→0, 2→1
            if orig_c == 0:
                c = 0
            elif orig_c == 2:
                c = 1
            else:
                continue  # skip any unexpected classes

            coords = list(map(float, parts[1:]))
            # de-normalize polygon points to pixel coords
            xs = [coords[i] * W for i in range(0, len(coords), 2)]
            ys = [coords[i] * H for i in range(1, len(coords), 2)]

            x1, x2 = min(xs), max(xs)
            y1, y2 = min(ys), max(ys)

            # convert to normalized YOLO bbox: x_center, y_center, width, height
            xc = ((x1 + x2) / 2) / W
            yc = ((y1 + y2) / 2) / H
            bw = (x2 - x1) / W
            bh = (y2 - y1) / H

            fout.write(f"{c} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

    print(f"[OK] wrote {out_path}")
