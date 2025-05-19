# import yolo
from ultralytics import YOLO
import cv2

# Load your trained model
model = YOLO("ball_detect/yolov8_balls5/weights/best.pt")

# video capture
cap = cv2.VideoCapture(0)  # 0 for default camera, or provide a video file path

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Predict on the current frame
    results = model.predict(source=frame, conf=0.25)

    # Visualize the results
    out = results[0].plot()  # NumPy array with boxes drawn
    cv2.imshow("Detections", out)

    # Exit on 'q' key press'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break