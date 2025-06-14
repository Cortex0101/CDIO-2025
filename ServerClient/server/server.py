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
from PID import *
from utility import *

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
        self.path_planner = PathPlanner(strategy=AStarStrategyOptimized(obj_radius=40))
        self.path_planner_visualizer = PathPlannerVisualizer()

        # if send_custom_instructions is true, we will be able to simply send instruction by entering
        # them in the console, otherwise we will have to use we will run the main_loop() method
        self.custom_instruction_loop()

        cv2.destroyAllWindows()

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
        # Increased Kp for more aggressive steering
        steer_pid = PID(Kp=3.0, Ki=0.0, Kd=0.2)
        speed_pid = PID(Kp=2, Ki=0.0, Kd=0.0)

        current_path = None
        following_path = False
        path_index = 0
        last_time = time.time()
        robot = None
        robot_direction = 0

        max_speed = 50  # Maximum wheel speed

        while True:
            ret, current_video_frame = self.cap.read()

            if not ret:
                print("[SERVER] Error: Could not read frame from camera.")
                continue

            self.course = self.ai_model.generate_course(current_video_frame)
            current_video_frame_with_objs = self.course_visualizer.draw(current_video_frame, self.course)

            key = cv2.waitKey(20) & 0xFF
            
            if key == ord('q'):
                print("[SERVER] Quitting...")
                break

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
                path_index = 0
                steer_pid.prev_error = 0
                steer_pid.integral = 0
                speed_pid.prev_error = 0
                speed_pid.integral = 0

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
                    robot_angle_rad = math.radians(robot_direction)

                    # Get next target point on path
                    target_point = get_next_path_point(robot_pos, current_path, lookahead=30) # upped from 20 to 30 and draw next point.
                    #current_video_frame_with_objs = self.path_planner_visualizer.draw_target_point(current_video_frame_with_objs, target_point)
                    cross_track_error = compute_distance(robot_pos, target_point)
                    heading_error = compute_angle_error(robot_pos, robot_angle_rad, target_point)

                    now = time.time()
                    dt = now - last_time
                    last_time = now

                    # If heading error is large, slow down or turn in place
                    if abs(heading_error) > math.radians(90):
                        speed = 2
                    elif abs(heading_error) > math.radians(45):
                        speed = 10
                    else:
                        speed = max(10, max_speed - speed_pid.update(abs(cross_track_error), dt))

                    steer = steer_pid.update(heading_error, dt)

                    # Clamp steer to avoid excessive turning
                    steer = max(-max_speed, min(max_speed, steer))

                    print(f"[SERVER] Robot pos: {robot_pos}, Target: {target_point}, Speed: {speed}, Steer: {steer}, Heading error: {math.degrees(heading_error):.2f} degrees")

                    # Send steer command to robot
                    self.send_instruction({
                        "cmd": "steer",
                        "speed": float(speed),
                        "steer": float(steer)
                    })

                    # Stop if close to goal
                    if compute_distance(robot_pos, current_path[-1]) < 15:
                        print("[SERVER] Reached goal, stopping.")
                        self.send_instruction({"cmd": "steer", "speed": 0, "steer": 0})
                        following_path = False
                        current_path = None

            cv2.imshow("view", current_video_frame_with_objs)
        


if __name__ == "__main__":
    server = Server(fakeEv3Connection=False)  # Set to True for fake EV3 connection