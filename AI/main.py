# import yolo
from ultralytics import YOLO
import cv2

def zoom_image(img, zoom_factor):
    """
    Crop the center of `img` by `zoom_factor` then resize back to original size.
    zoom_factor > 1.0 zooms in; e.g. 1.5 → 150% zoom.
    """
    h, w = img.shape[:2]
    # compute crop w,h
    new_w, new_h = int(w/zoom_factor), int(h/zoom_factor)
    x1 = (w - new_w)//2
    y1 = (h - new_h)//2
    cropped = img[y1:y1+new_h, x1:x1+new_w]
    # scale back to (w,h)
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

# Load your trained model
model = YOLO("ball_detect/v3_balls_s_night_run/weights/best.pt")

img = cv2.imread("AI/datasets/V3/test/images/CustomImage-2023-06-16_143153_jpg.rf.684f8d04f1f508e9caffac147b6c60b6.jpg")

# create 10% zoom, 20% zoom, 30% zoom, 40% zoom, 50% zoom
zoomed_images = []
for i in range(1, 30):
    zoomed_images.append(zoom_image(img, 1 + i * 0.1))

# white ball confidence list
white_ball_confidence = []
orange_ball_confidence = []

# show predictions on each zoomed image
for i, zoomed_img in enumerate(zoomed_images):
    # Predict on the current frame
    results = model.predict(source=zoomed_img, conf=0.01)

    # Visualize the results
    out = results[0].plot()  # NumPy array with boxes drawn

    # Show the image with predictions
    cv2.imshow(f"Zoomed Image {i+1}", out)
    cv2.waitKey(0)  # Wait for a key press to close the window

    # Get the confidence scores for white and orange balls
    for result in results[0].boxes.data:
        class_id = int(result[5])
        confidence = result[4]
        if class_id == 0:  # Assuming 0 is the class ID for white balls
            white_ball_confidence.append(confidence)
        elif class_id == 1:  # Assuming 1 is the class ID for orange balls
            orange_ball_confidence.append(confidence)

# Print the white ball confidence scores tensor(0.8414, device='cuda:0')
print("White Ball Confidence Scores:", white_ball_confidence.__str__().replace("tensor(", "").replace(", device='cuda:0')", ""))
#yolo detect train data=AI/datasets/V3/data.yml model=yolov8s.pt device=0 imgsz=640 batch=8 epochs=50 augment=True multi_scale=True project=ball_detect name=v3_balls_s_aug_final
#D:\CDIO25\CDIO-2025> yolo detect train data=AI/datasets/V3/data.yml model=yolov8s.pt device=0 imgsz=640 batch=8 epochs=50 augment=True multi_scale=True project=ball_detect name=v3_balls_s_night_run