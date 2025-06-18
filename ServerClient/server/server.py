import socket
import json
import time
import math
import sys
import select
import threading
import queue
from enum import Enum

import cv2

from AImodel import *
from Course import *
from PathPlanner import *
from PurePursuit import *

def distance(a, b):
    return math.hypot(b[0] - a[0], b[1] - a[1])

def angle_to(src, dst):
    return math.degrees(math.atan2(dst[1] - src[1], dst[0] - src[0]))

class RobotState(Enum):
    IDLE = 0
    FOLLOW_PATH = 1
    TURN_TO_OBJECT = 2

class Server:
    def __init__(self, fakeEv3Connection=False):
        self.host = '0.0.0.0'
        self.port = 12346
        self.SEND_CUSTOM_INSTRUCTIONS = True
        self.CONTROL_CUSTOM = False

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

        self.ai_model = AIModel("ball_detect/v12/weights/best.pt")
        self.course = Course()
        self.course_visualizer = CourseVisualizer(draw_boxes=True, draw_labels=True, draw_confidence=True, draw_masks=False)
        self.path_planner = PathPlanner(strategy=AStarStrategyOptimized(obj_radius=50))
        self.path_planner_visualizer = PathPlannerVisualizer()

        # extra 
        self.pure_pursuit_navigator = PurePursuitNavigator(None, 
                                                lookahead_distance=25, 
                                                max_speed=25, 
                                                true_max_speed=25, 
                                                kp=0.75, 
                                                max_turn_slowdown=1)
        self.robot_state = RobotState.IDLE
        self.robot = None           # For storing previous robot if needed
        self.robot_direction = 0    # For storing previous robot direction if needed

        # if send_custom_instructions is true, we will be able to simply send instruction by entering
        # them in the console, otherwise we will have to use we will run the main_loop() method
        if self.SEND_CUSTOM_INSTRUCTIONS:
            self.custom_instruction_loop()
        elif self.CONTROL_CUSTOM:
            self.control_custom_loop()

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

    def main_loop(self):
        while True:
            # Read the current video frame
            ret, current_video_frame = self.cap.read()
            self.course = self.ai_model.generate_course(current_video_frame)
            processed_img = self.course_visualizer.draw(current_video_frame, self.course)

            # keyboard input handling
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("[SERVER] Quitting...")
                break
            elif key == ord('c'):
                self.purse_pursuit_navigator.set_path(None)
                self.robot_state = RobotState.IDLE
                instruction = {"cmd": "drive", "left_speed": 0, "right_speed": 0}
                self.send_instruction(instruction)

            # Main algorithm
            if self.robot_state == RobotState.IDLE:
                # Work from balls closest to small goal?
                pass
            elif self.robot_state == RobotState.FOLLOW_PATH:
                pass
            

    def control_custom_loop(self):
        last_instruction =  {"cmd": "drive", "left_speed": 0, "right_speed": 0}

        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("[SERVER] Quitting...")
                break

            if key == ord('e'):
                last_instruction = {"cmd": "drive", "left_speed": 24, "right_speed": 24}
                print(last_instruction)
                self.send_instruction(last_instruction)
            elif key == ord('s'):
                last_instruction = {"cmd": "drive", "left_speed": -24, "right_speed": 24}
                print(last_instruction)
                self.send_instruction(last_instruction)
            elif key == ord('d'):
                last_instruction = {"cmd": "drive", "left_speed": -24, "right_speed": -24}
                print(last_instruction)
                self.send_instruction(last_instruction)
            elif key == ord('f'):
                last_instruction = {"cmd": "drive", "left_speed": 24, "right_speed": -24}
                print(last_instruction)
                self.send_instruction(last_instruction)
            elif key == ord('r'):
                last_instruction = {"cmd": "drive", "left_speed": 0, "right_speed": 0}
                print(last_instruction)
                self.send_instruction(last_instruction)


            # show empty frame
            cv2.imshow("view",  np.zeros((480, 640, 3), dtype=np.uint8))

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
        current_state = RobotState.IDLE
        purse_pursuit_navigator = PurePursuitNavigator(None, 
                                                        lookahead_distance=25, 
                                                        max_speed=25, 
                                                        true_max_speed=25, 
                                                        kp=0.75, 
                                                        max_turn_slowdown=1)
        robot = None
        robot_direction = 0
        angle_to_target = -1 # used when in RobotState.TURN_TO_OBJECT

        while True:
            ret, current_video_frame = self.cap.read()

            if not ret:
                print("[SERVER] Error: Could not read frame from camera.")
                continue

            self.course = self.ai_model.generate_course(current_video_frame)
            current_video_frame_with_objs = self.course_visualizer.draw(current_video_frame, self.course)

            # robot and direction
            if self.course.get_robot() is None:
                print("[SERVER] No robot detected in course, using previous position.")
            else:
                robot = self.course.get_robot()
                if robot.direction is None:
                    print("[SERVER] Robot direction is None, using previous value.")
                else:
                    robot_direction = robot.direction

            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("[SERVER] Quitting...")
                break
            elif key == ord('c'):
                current_state = RobotState.IDLE
                purse_pursuit_navigator.set_path(None)
                instruction = {"cmd": "drive", "left_speed": 0, "right_speed": 0}
                self.send_instruction(instruction)
            # if key is number 1
            elif key == ord('1'):
                current_state = RobotState.FOLLOW_PATH
                purse_pursuit_navigator.set_path(None)
                instruction = {"cmd": "drive", "left_speed": 0, "right_speed": 0}
                self.send_instruction(instruction)
            elif key == ord('2'):
                current_state = RobotState.TURN_TO_OBJECT
                purse_pursuit_navigator.set_path(None)
                instruction = {"cmd": "drive", "left_speed": 0, "right_speed": 0}
                self.send_instruction(instruction)

            # Handle mouse click to generate path
            if self.mouse_clicked_coords[0] is not None:
                x, y = self.mouse_clicked_coords[0]
                self.mouse_clicked_coords[0] = None
                
                if current_state == RobotState.FOLLOW_PATH:
                    grid = self.path_planner.generate_grid(self.course, True) # change to True if you want to drw floor
                    grid_img = self.path_planner_visualizer.draw_grid_objects(grid)
                    start = robot.center
                    start = (int(start[0]), int(start[1]))
                    end = (int(x), int(y))
                    print(f"[SERVER] Generating path from {start} to {end}...")
                    current_path = self.path_planner.find_path(start, end, grid)
                    print(f"[SERVER] Path found: {len(current_path)} points.")
                    cv2.imshow("grid_visualization", grid_img)
                elif current_state == RobotState.TURN_TO_OBJECT:
                    closest_obj = self.course.get_nearest_object((x, y))
                    if closest_obj is not None:
                        dst_point = closest_obj.center
                        src_point = robot.center
                        print(f"[SERVER] Turning to object {closest_obj} at {dst_point} from {src_point}...")
                        angle_to_target = angle_to(src_point, dst_point)

            # If currently following a path
            if purse_pursuit_navigator.path is not None and current_state == RobotState.FOLLOW_PATH:
                current_video_frame_with_objs = self.path_planner_visualizer.draw_path(current_video_frame_with_objs, current_path)       
                instruction = purse_pursuit_navigator.compute_drive_command(robot.center, robot_direction)
                self.send_instruction(instruction)
                if distance(robot.center, current_path[-1]) < 30:
                    print("[SERVER] Reached the end of the path.")
                    purse_pursuit_navigator.set_path(None)
                    instruction = {"cmd": "drive", "left_speed": 0, "right_speed": 0}
                    self.send_instruction(instruction)
            elif current_state == RobotState.TURN_TO_OBJECT and angle_to_target != -1:
                # Compute turn command to face the target object
                instruction = purse_pursuit_navigator.compute_turn_command(robot_direction, angle_to_target)
                self.send_instruction(instruction)

                # Check if the robot is facing the target object
                if abs(angle_to_target - robot_direction) < 5:
                    print("[SERVER] Robot is now facing the target object.")
                    current_state = RobotState.IDLE
                    angle_to_target = -1  # Reset angle to target


            # draw current state string in top left of the frame
            cv2.putText(current_video_frame_with_objs, f"State: {current_state.name}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow("view", current_video_frame_with_objs)

    def custom_instruction_loop_old(self):
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
                
                grid = self.path_planner.generate_grid(self.course, True) # change to True if you want to drw floor
                grid_img = self.path_planner_visualizer.draw_grid_objects(grid)

                start = self.course.get_robot().center
                start = (int(start[0]), int(start[1]))
                end = (int(x), int(y))
                print(f"[SERVER] Generating path from {start} to {end}...")

                current_path = self.path_planner.find_path(start, end, grid)
                print(f"[SERVER] Path found: {len(current_path)} points.")
                cv2.imshow("grid_visualization", grid_img)
                following_path = True  # Start following the path
                '''
                purse_pursuit_navigator = PurePursuitNavigator(current_path, 
                                                               lookahead_distance=25, 
                                                               max_speed=20, 
                                                               true_max_speed=20, 
                                                               kp=0.6, 
                                                               max_turn_slowdown=1)
                '''
                purse_pursuit_navigator = PurePursuitNavigator(current_path, 
                                                               lookahead_distance=25, 
                                                               max_speed=25, 
                                                               true_max_speed=25, 
                                                               kp=0.75, 
                                                               max_turn_slowdown=1)

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