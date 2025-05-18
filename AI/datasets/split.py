import os, glob, random, shutil

src_img = 'AI/datasets/images'
src_lbl = 'AI/datasets/labels_yolo'  # your converted YOLO‐bbox labels
for split in ('train','val'):
    os.makedirs(f'AI/datasets/{split}/images', exist_ok=True)
    os.makedirs(f'AI/datasets/{split}/labels', exist_ok=True)

files = glob.glob(os.path.join(src_img,"*.jpg"))
random.shuffle(files)
n_train = int(len(files)*0.8)

for i, img in enumerate(files):
    base = os.path.basename(img)
    lbl = os.path.join(src_lbl, base.replace('.jpg','.txt'))
    subset = 'train' if i < n_train else 'val'
    shutil.copy(img,  f'AI/datasets/{subset}/images/{base}')
    shutil.copy(lbl,  f'AI/datasets/{subset}/labels/{os.path.basename(lbl)}')