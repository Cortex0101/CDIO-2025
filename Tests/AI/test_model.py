import pytest 
from ultralytics import YOLO
import cv2

img_one_path = "ressources/img/3balls_home.jpg"
img_two_path = "ressources/img/5balls_normal_view.jpg"
img_three_path = "ressources/img/5balls_obscure_view.jpg"

# Load your trained model
model = YOLO("ball_detect/yolov8_balls5/weights/best.pt")

def test_image_one():
    # Predict on one image
    results = model.predict(
        source="ressources/img/3balls_home.jpg",
        imgsz=640,
        conf=0.25,
        save=True  # saves to runs/detect/predict
    )

    # 3 balls
    assert len(results[0].boxes) == 3, f"Expected 3 balls, but got {len(results[0].boxes)}"	

def test_image_two():
    # Predict on one image
    results = model.predict(
        source="ressources/img/5balls_normal_view.jpg",
        imgsz=640,
        conf=0.25,
        save=True  # saves to runs/detect/predict
    )

    # 1 ball
    assert len(results[0].boxes) == 5, f"Expected 5 ball, but got {len(results[0].boxes)}"

def test_image_three():
    # Predict on one image
    results = model.predict(
        source="ressources/img/5balls_obscure_view.jpg",
        imgsz=640,
        conf=0.25,
        save=True  # saves to runs/detect/predict
    )

    # 1 ball
    assert len(results[0].boxes) == 5, f"Expected 5 ball, but got {len(results[0].boxes)}"
