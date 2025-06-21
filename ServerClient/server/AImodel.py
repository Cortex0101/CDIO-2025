from ultralytics import YOLO
import numpy as np
import math
import logging

import cv2

from Course import Course, CourseObject

logger = logging.getLogger(__name__)

class AIModel:
    def __init__(self, model_path: str, min_confidence: float = 0.5):
        logger.info(f"Loading AI model from {model_path} with min confidence {min_confidence}")
        logger.info(f"Using CUDA: {cv2.cuda.getCudaEnabledDeviceCount() > 0}")
        self.min_confidence = min_confidence
        self.model = YOLO(model_path)

    def generate_course(self, source):
        results = self.model.predict(source=source, verbose=False, conf=self.min_confidence, device=0, imgsz=640)
        return Course.from_yolo_results(results[0])
    
    def predict_stream(self):
        return self.model.predict(
            source=0,              # webcam
            stream=True,          # enables generator
            conf=self.min_confidence,
            imgsz=640,            # faster
            device=0,         # since no CUDA
            verbose=False
        )