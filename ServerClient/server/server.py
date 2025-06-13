import socket
import json
import time
import math
import sys
import select
import threading
import queue

import cv2

from AImodel import *
from Course import *
from PathPlanner import *

class Server:
    def __init__(self):
        self.host = '0.0.0.0'
        self.port = 12346
        self.SEND_CUSTOM_INSTRUCTIONS = True

        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.bind((self.host, self.port))
        self.server.listen(1)
        print(f"[SERVER] Listening on {self.host}:{self.port}... Waiting for EV3 connection.")
        self.conn, addr = self.server.accept()
        print(f"[SERVER] Connected to EV3 at {addr}")

        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            print("[SERVER] Error: Could not open camera.")
            exit()

        self.mouse_clicked_coords = [None]
        cv2.namedWindow("view")
        cv2.setMouseCallback("view", self.mouse_callback)

        self.ai_model = AIModel("ball_detect/v8/weights/best.pt")
        self.course = Course()
        self.course_visualizer = CourseVisualizer(draw_boxes=True, draw_labels=True, draw_masks=False)
        self.path_planner = PathPlanner(strategy=AStarStrategyOptimized(obj_radius=10))
        self.path_planner_visualizer = None

        # if send_custom_instructions is true, we will be able to simply send instruction by entering
        # them in the console, otherwise we will have to use we will run the main_loop() method
        self.custom_instruction_loop()

    def main_loop():
        while True:
             # do nothing for now
            pass

    def send_instruction(self, instruction, wait_for_response=False):
        try:
            self.conn.sendall(json.dumps(instruction).encode())
            if wait_for_response:
                data = self.conn.recv(1024)
                response = json.loads(data.decode())
                return response
        except Exception as e:
            print(f"[SERVER] Error sending instruction: {e}")
            return {"status": "error", "msg": str(e)}

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"[SERVER] Mouse clicked at ({x}, {y})")
            self.mouse_clicked_coords[0] = (x, y)

    def custom_instruction_loop(self):
        while True:
            ret, frame = self.cap.read()

            if not ret:
                print("[SERVER] Error: Could not read frame from camera.")
                continue

            self.course = self.ai_model.generate_course(frame)
            img = self.course_visualizer.draw(frame, self.course)
            cv2.imshow("view", img)
            cv2.waitKey(0)



if __name__ == "__main__":
    server = Server()