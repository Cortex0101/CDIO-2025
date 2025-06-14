from ultralytics import YOLO
import numpy as np
import math
import cv2
from Course import Course, CourseObject

class AIModel:
    def __init__(self, model_path: str, min_confidence: float = 0.5):
        self.min_confidence = min_confidence
        try:
            self.model = YOLO(model_path)
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            self.model = None

    def generate_course(self, source):
        if self.model is None:
            raise RuntimeError("YOLO model not loaded.")
        results = self.model.predict(source=source, verbose=False, conf=self.min_confidence)
        return Course.from_yolo_results(results[0])