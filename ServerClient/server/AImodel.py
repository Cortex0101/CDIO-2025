from ultralytics import YOLO
import numpy as np
import math
import cv2
from Course import Course, CourseObject

class AIModel:
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)

    def generate_course(self, source):
        """
        Generate a Course from an image or video source.

        Args:
            source: np.ndarray or image file path
        Returns:
            Course: detected objects in the frame
        """
        results = self.model.predict(source=source, verbose=False)
        course = Course.from_yolo_results(results[0])
        return course
