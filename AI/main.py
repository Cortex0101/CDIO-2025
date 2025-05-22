# import yolo
from ultralytics import YOLO
import cv2

# Load your trained model
model = YOLO("ball_detect/v3_balls3/weights/best.pt")

# Predict on one image
results = model.predict(
    source="ressources/img/3balls_home.jpg",
    imgsz=640,
    conf=0.25,
    save=True  # saves to runs/detect/predict
)

# Or visualize in memory
img = cv2.imread("ressources/img/3balls_home.jpg")
out = results[0].plot()         # NumPy array with boxes drawn
cv2.imshow("Detections", out)
cv2.waitKey(0)