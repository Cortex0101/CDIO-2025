import os, glob, random
import cv2

IMG_DIR    = "AI/datasets/images"
LBL_DIR    = "AI/datasets/labels"
NEG_DIR    = "AI/datasets/negatives"
NB_NEG     = 500        # how many negative crops you want
CROP_SIZE  = (64, 64)   # must match your cascade’s -w and -h

os.makedirs(NEG_DIR, exist_ok=True)

# 1. load all your bounding-boxes into memory
from collections import defaultdict
gt = defaultdict(list)
for lbl in glob.glob(os.path.join(LBL_DIR, "*.txt")):
    base = os.path.splitext(os.path.basename(lbl))[0]
    img_w, img_h = cv2.imread(os.path.join(IMG_DIR, base+".jpg")).shape[1], cv2.imread(os.path.join(IMG_DIR, base+".jpg")).shape[0]
    for line in open(lbl):
        parts = list(map(float, line.split()[1:]))
        xs = [parts[i]*img_w   for i in range(0,len(parts),2)]
        ys = [parts[i]*img_h   for i in range(1,len(parts),2)]
        x1, x2 = int(min(xs)), int(max(xs))
        y1, y2 = int(min(ys)), int(max(ys))
        gt[base].append((x1,y1,x2,y2))

# 2. sample random crops
neg_list = []
for i in range(NB_NEG):
    img_path = random.choice(glob.glob(os.path.join(IMG_DIR, "*.jpg")))
    name = os.path.splitext(os.path.basename(img_path))[0]
    img = cv2.imread(img_path)
    h,w = img.shape[:2]

    # try until you find a crop that doesn’t intersect ANY gt-box
    while True:
        x = random.randint(0, w-CROP_SIZE[0])
        y = random.randint(0, h-CROP_SIZE[1])
        crop_rect = (x, y, x+CROP_SIZE[0], y+CROP_SIZE[1])
        if all( not (x1 < crop_rect[2] and x2 > crop_rect[0] and
                     y1 < crop_rect[3] and y2 > crop_rect[1])
                for (x1,y1,x2,y2) in gt[name] ):
            neg_img = img[y:y+CROP_SIZE[1], x:x+CROP_SIZE[0]]
            out_name = f"neg_{i}.jpg"
            cv2.imwrite(os.path.join(NEG_DIR, out_name), neg_img)
            neg_list.append(os.path.join("negatives", out_name))
            break

# 3. write negatives.txt
with open("AI/datasets/negatives.txt","w") as f:
    f.write("\n".join(neg_list))
