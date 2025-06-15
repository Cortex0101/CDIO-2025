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
from PurePursuit import *

def distance(a, b):
    return math.hypot(b[0] - a[0], b[1] - a[1])

class Server:
    def __init__(self, fakeEv3Connection=False):
        self.host = '0.0.0.0'
        self.port = 12346
        self.SEND_CUSTOM_INSTRUCTIONS = True

        if fakeEv3Connection:
            print("[SERVER] Fake EV3 connection enabled. No actual socket connection will be established.")
            self.conn = None
        else: 
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
        cv2.namedWindow("grid_visualization")

        self.ai_model = AIModel("ball_detect/v8/weights/best.pt")
        self.course = Course()
        self.course_visualizer = CourseVisualizer(draw_boxes=True, draw_labels=True, draw_confidence=True, draw_masks=False)
        self.path_planner = PathPlanner(strategy=AStarStrategyOptimized(obj_radius=50))
        self.path_planner_visualizer = PathPlannerVisualizer()

        # if send_custom_instructions is true, we will be able to simply send instruction by entering
        # them in the console, otherwise we will have to use we will run the main_loop() method
        self.custom_instruction_loop()

        cv2.destroyAllWindows()

    def accept_connections_loop(self):
        if self.fakeEv3Connection:
            print("[SERVER] Fake EV3 connection enabled. No actual socket connection will be established.")
            self.conn = None
            self.custom_instruction_loop()
            return

        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind((self.host, self.port))
        self.server.listen(1)
        print(f"[SERVER] Listening on {self.host}:{self.port}... Waiting for EV3 connection.")

        while True:
            print("[SERVER] Waiting for a new EV3 connection...")
            self.conn, addr = self.server.accept()
            print(f"[SERVER] Connected to EV3 at {addr}")
            try:
                self.custom_instruction_loop()
            except Exception as e:
                print(f"[SERVER] Connection lost or error: {e}")
            finally:
                try:
                    self.conn.close()
                except Exception:
                    pass
                print("[SERVER] Connection closed.")

    def main_loop():
        while True:
             # do nothing for now
            pass

    def send_instruction(self, instruction, wait_for_response=False):
        try:
            self.conn.sendall((json.dumps(instruction) + '\n').encode())
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
        current_path = None
        following_path = False
        robot = None
        robot_direction = 0
        purse_pursuit_navigator = None

        while True:
            ret, current_video_frame = self.cap.read()

            if not ret:
                print("[SERVER] Error: Could not read frame from camera.")
                continue

            self.course = self.ai_model.generate_course(current_video_frame)
            current_video_frame_with_objs = self.course_visualizer.draw(current_video_frame, self.course)

            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("[SERVER] Quitting...")
                break

            if key == ord('c'):
                following_path = False
                current_path = None
                purse_pursuit_navigator = None
                instruction = {"cmd": "drive", "left_speed": 0, "right_speed": 0}
                self.send_instruction(instruction)

            # Handle mouse click to generate path
            if self.mouse_clicked_coords[0] is not None:
                x, y = self.mouse_clicked_coords[0]
                self.mouse_clicked_coords[0] = None
                
                grid = self.path_planner.generate_grid(self.course, True)
                grid_img = self.path_planner_visualizer.draw_grid_objects(grid)

                start = self.course.get_robot().center
                start = (int(start[0]), int(start[1]))
                end = (int(x), int(y))
                print(f"[SERVER] Generating path from {start} to {end}...")

                current_path = self.path_planner.find_path(start, end, grid)
                print(f"[SERVER] Path found: {len(current_path)} points.")
                cv2.imshow("grid_visualization", grid_img)
                following_path = True  # Start following the path
                purse_pursuit_navigator = PurePursuitNavigator(current_path, lookahead_distance=20, max_speed=75, true_max_speed=10)

            # Draw path if exists
            if current_path is not None:
                current_video_frame_with_objs = self.path_planner_visualizer.draw_path(current_video_frame_with_objs, current_path)

            # Path following logic
            if following_path and current_path is not None and len(current_path) > 1:
                if self.course.get_robot() is None:
                    print("[SERVER] No robot detected in course, use previous position.")
                else: 
                    robot = self.course.get_robot()
                if robot is not None:
                    robot_pos = robot.center
                    if robot.direction is not None:
                        robot_direction = robot.direction
                    else:
                        print("[SERVER] Robot direction is None, use previous value.")
                    
                instruction = purse_pursuit_navigator.compute_drive_command(robot_pos, robot_direction)
                self.send_instruction(instruction)

                if distance(robot_pos, current_path[-1]) < 30:
                    print("[SERVER] Reached the end of the path.")
                    following_path = False
                    current_path = None
                    purse_pursuit_navigator = None
                    instruction = {"cmd": "drive", "left_speed": 0, "right_speed": 0}
                    self.send_instruction(instruction)

            cv2.imshow("view", current_video_frame_with_objs)

if __name__ == "__main__":
    server = Server(fakeEv3Connection=False)  # Set to True for fake EV3 connection