import cv2
import os
import numpy as np
from glob import glob

FOLDER = "AI/images"

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

zoom_factor = 1.0
offset = np.array([0.0, 0.0])
cursor_pos = (0, 0)
is_panning = False
pan_start = (0, 0)
offset_start = (0, 0)

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
        for line in f:
            values = line.strip().split()
            cls = int(values[0])
            coords = list(map(float, values[3:]))
            points = [(int(coords[i] * img_w), int(coords[i + 1] * img_h)) for i in range(0, len(coords), 2)]
            polys.append({"class_id": cls, "points": points})
    return polys

def point_in_polygon(point, polygon):
    return cv2.pointPolygonTest(np.array(polygon, np.int32), point, False) >= 0

def screen_to_image_coords(x, y):
    # Calculate image-space coordinates from screen-space, adjusted for zoom and offset
    view_x = int(x + offset[0])
    view_y = int(y + offset[1])
    ix = int(view_x / zoom_factor)
    iy = int(view_y / zoom_factor)
    ix = np.clip(ix, 0, image.shape[1] - 1)
    iy = np.clip(iy, 0, image.shape[0] - 1)
    return ix, iy

def draw_polygons(img, polys):
    overlay = img.copy()
    for poly in polys:
        pts = np.array(poly["points"], np.int32)
        color = CLASS_COLORS[poly["class_id"]]
        if len(pts) >= 3:
            cv2.fillPoly(overlay, [pts], color)
        cv2.polylines(img, [pts], True, color, 2)
        for (x, y) in poly["points"]:
            cv2.circle(img, (x, y), 3, color, -1)
        if len(pts) > 0:
            cv2.putText(img, CLASSES[poly["class_id"]], pts[0], cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)

def draw_current_polygon(img, points):
    if points:
        cv2.polylines(img, [np.array(points)], False, (200, 200, 200), 1)
        for (x, y) in points:
            cv2.circle(img, (x, y), 2, (100, 100, 100), -1)

def show_polygon_mask(image, polygon):
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, [np.array(polygon, np.int32)], (255, 255, 255))
    result = cv2.bitwise_and(image, mask)
    return result

def show_all_polygon_masks(image, polys):
    mask = np.zeros_like(image)
    for poly in polys:
        temp_mask = np.zeros_like(image)
        cv2.fillPoly(temp_mask, [np.array(poly["points"], np.int32)], (255, 255, 255))
        masked = cv2.bitwise_and(image, temp_mask)
        mask = cv2.add(mask, masked)
    cv2.imshow("All Polygon Masks", mask)

def print_help():
    print("""
=== Shortcut Guide ===
[Mouse Controls]
  Left click        - Add point to current polygon
  Right click       - Delete polygon under cursor
  Middle click drag - Pan the view
  Mouse wheel       - Zoom toward cursor
  Double middle click - Show mask of polygon under cursor

[Keyboard Controls]
  1–8               - Switch class
  Enter             - Save current polygon
  x                 - Cancel current polygon
  z                 - Undo last polygon
  m                 - Next image
  n                 - Previous image
  space             - Show all polygon masks
  h                 - Show this help menu
  q                 - Save and quit
======================
          
NOTE: Hvis du zoomer og klikker på kanten af billedet, er det offset. Zoom ud til normal størrelse for at undgå dette.
""")

def mouse_callback(event, x, y, flags, param):
    global current_polygon, polygons, zoom_factor, offset, cursor_pos
    global is_panning, pan_start, offset_start

    cursor_pos = (x, y)
    zx, zy = screen_to_image_coords(x, y)

    if event == cv2.EVENT_LBUTTONDOWN and not is_panning:
        current_polygon.append((zx, zy))

    if event == cv2.EVENT_MBUTTONDBLCLK and not is_panning:
        for i, poly in enumerate(polygons):
            if point_in_polygon((zx, zy), poly["points"]):
                result = show_polygon_mask(image, poly["points"])
                cv2.imshow("Polygon Mask", result)
                break

    elif event == cv2.EVENT_RBUTTONDOWN and not is_panning:
        for i, poly in enumerate(polygons):
            if point_in_polygon((zx, zy), poly["points"]):
                polygons.pop(i)
                break

    elif event == cv2.EVENT_MOUSEWHEEL:
        old_zoom = zoom_factor
        zoom_step = 1.1 if flags > 0 else 0.9
        zoom_factor *= zoom_step
        zoom_factor = max(0.2, min(zoom_factor, 10))

        dx = cursor_pos[0] * (1 / old_zoom)
        dy = cursor_pos[1] * (1 / old_zoom)
        new_dx = cursor_pos[0] * (1 / zoom_factor)
        new_dy = cursor_pos[1] * (1 / zoom_factor)
        offset[0] += (dx - new_dx) * zoom_factor
        offset[1] += (dy - new_dy) * zoom_factor

    elif event == cv2.EVENT_MBUTTONDOWN:
        is_panning = True
        pan_start = (x, y)
        offset_start = offset.copy()

    elif event == cv2.EVENT_MOUSEMOVE and is_panning:
        dx = x - pan_start[0]
        dy = y - pan_start[1]
        offset[0] = offset_start[0] - dx
        offset[1] = offset_start[1] - dy

    elif event == cv2.EVENT_MBUTTONUP:
        is_panning = False

def load_current_image():
    global image, polygons
    img_path = images[index]
    image = load_image(img_path)
    label_path = os.path.splitext(img_path)[0] + ".txt"
    polygons.clear()
    polygons.extend(load_labels(label_path, image.shape[1], image.shape[0]))
    print(f"Loaded image: {img_path} with {len(polygons)} polygons.")

def save_current_labels():
    img_path = images[index]
    label_path = os.path.splitext(img_path)[0] + ".txt"
    save_labels(label_path, polygons, image.shape[1], image.shape[0])

def run_labeler():
    global index, current_class, current_polygon, offset

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

        zoomed = cv2.resize(temp, None, fx=zoom_factor, fy=zoom_factor)
        h, w = image.shape[:2]
        zh, zw = zoomed.shape[:2]

        x1 = int(np.clip(offset[0], 0, max(zw - w, 0)))
        y1 = int(np.clip(offset[1], 0, max(zh - h, 0)))
        x2 = min(x1 + w, zw)
        y2 = min(y1 + h, zh)

        view = zoomed[y1:y2, x1:x2]
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        canvas[0:view.shape[0], 0:view.shape[1]] = view

        cv2.putText(canvas, f"Class: {CLASSES[current_class]} ({current_class})", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("Segment Tool", canvas)

        key = cv2.waitKey(50) & 0xFF

        if key in [ord(str(i + 1)) for i in range(len(CLASSES))]:
            current_class = key - ord('1')
        elif key == ord('z') and polygons:
            polygons.pop()
        elif key == ord('x'):
            current_polygon.clear()
        elif key == 13 and len(current_polygon) >= 3:
            polygons.append({"class_id": current_class, "points": current_polygon.copy()})
            current_polygon.clear()
        elif key == ord('m'):
            save_current_labels()
            index = (index + 1) % len(images)
            offset[:] = 0, 0
            load_current_image()
        elif key == ord('n'):
            save_current_labels()
            index = (index - 1) % len(images)
            offset[:] = 0, 0
            load_current_image()
        elif key == ord(' '):
            show_all_polygon_masks(image, polygons)
        elif key == ord('h'):
            print_help()
        elif key == ord('q'):
            save_current_labels()
            break

    cv2.destroyAllWindows()

run_labeler()
