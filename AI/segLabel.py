import cv2
import os
import numpy as np
from glob import glob

# === CONFIGURATION ===
FOLDER = "AI/images"  # both .jpg and .txt are stored here

CLASSES = [
    "orange_ball", "white_ball", "egg", "cross",
    "robot", "small_goal", "large_goal", "wall"
]

CLASS_COLORS = [
    (255, 128, 0), (255, 255, 255), (255, 255, 0), (255, 0, 255),
    (0, 255, 255), (0, 255, 0), (0, 128, 255), (128, 128, 128)
]

images = sorted(glob(os.path.join(FOLDER, "*.jpg")))
index = 0
image = None
polygons = []
current_class = 0
current_polygon = []

def load_image(path):
    return cv2.imread(path)

def save_labels(path, polys, img_w, img_h):
    with open(path, "w") as f:
        for item in polys:
            cls = item["class_id"]
            points = item["points"]
            cx = np.mean([p[0] for p in points]) / img_w
            cy = np.mean([p[1] for p in points]) / img_h
            poly_flat = [f"{x / img_w:.6f} {y / img_h:.6f}" for (x, y) in points]
            line = f"{cls} {cx:.6f} {cy:.6f} " + " ".join(poly_flat)
            f.write(line + "\n")

def load_labels(path, img_w, img_h):
    if not os.path.exists(path):
        return []
    polys = []
    with open(path) as f:
        for line in f.readlines():
            values = line.strip().split()
            cls = int(values[0])
            points = list(map(float, values[3:]))
            point_pairs = [(int(points[i] * img_w), int(points[i + 1] * img_h)) for i in range(0, len(points), 2)]
            polys.append({"class_id": cls, "points": point_pairs})
    return polys

def draw_polygons(img, poly_list):
    overlay = img.copy()
    for poly in poly_list:
        color = CLASS_COLORS[poly["class_id"] % len(CLASS_COLORS)]
        pts = np.array(poly["points"], np.int32)
        if len(pts) >= 3:
            cv2.fillPoly(overlay, [pts], color=color)
        cv2.polylines(img, [pts], isClosed=True, color=color, thickness=2)
        if len(pts) > 0:
            cv2.putText(img, CLASSES[poly["class_id"]], pts[0], cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        for (x, y) in poly["points"]:
            cv2.circle(img, (x, y), 3, color, -1)
    cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)

def draw_current_polygon(img, points):
    if len(points) >= 1:
        cv2.polylines(img, [np.array(points)], isClosed=False, color=(200, 200, 200), thickness=1)
        for (x, y) in points:
            cv2.circle(img, (x, y), 2, (100, 100, 100), -1)

def mouse_callback(event, x, y, flags, param):
    global current_polygon
    if event == cv2.EVENT_LBUTTONDOWN:
        current_polygon.append((x, y))

def load_current_image():
    global image, polygons
    img_path = images[index]
    image = load_image(img_path)
    label_path = os.path.splitext(img_path)[0] + ".txt"
    polygons.clear()
    polygons.extend(load_labels(label_path, image.shape[1], image.shape[0]))

def save_current_labels():
    img_path = images[index]
    label_path = os.path.splitext(img_path)[0] + ".txt"
    save_labels(label_path, polygons, image.shape[1], image.shape[0])

def run_segmentation_labeler():
    global index, current_class, current_polygon

    if not images:
        print("No images found.")
        return

    cv2.namedWindow("Segment Tool")
    cv2.setMouseCallback("Segment Tool", mouse_callback)
    load_current_image()

    while True:
        temp = image.copy()
        draw_polygons(temp, polygons)
        draw_current_polygon(temp, current_polygon)

        cv2.putText(temp, f"Class: {CLASSES[current_class]} ({current_class})", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("Segment Tool", temp)
        key = cv2.waitKey(50) & 0xFF

        if key in [ord(str(i + 1)) for i in range(min(9, len(CLASSES)))]:
            current_class = key - ord('1')
        elif key == ord('z') and polygons:
            polygons.pop()
        elif key == ord('x'):
            current_polygon.clear()
        elif key == 13:  # Enter key
            if len(current_polygon) >= 3:
                polygons.append({"class_id": current_class, "points": current_polygon.copy()})
            current_polygon.clear()
        elif key == ord('m'):
            save_current_labels()
            index = (index + 1) % len(images)
            load_current_image()
        elif key == ord('n'):
            save_current_labels()
            index = (index - 1) % len(images)
            load_current_image()
        elif key == ord('q'):
            save_current_labels()
            break

    cv2.destroyAllWindows()

run_segmentation_labeler()
