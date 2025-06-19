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
import keyboard

from AImodel import *
from Course import *
from PathPlanner import *
from PurePursuit import *
import config

def distance(a, b):
    return math.hypot(b[0] - a[0], b[1] - a[1])

def angle_to(src, dst):
    # clamped to 0-360 where 0 is right, 90 is up, 180 is left, 270 is down
    angle = math.degrees(math.atan2(dst[1] - src[1], dst[0] - src[0]))
    if angle < 0:
        angle += 360
    return angle  # returns angle in degrees, 0 is right, 90 is up, 180 is left, 270 is down

class RobotState(Enum):
    IDLE = 0
    FOLLOW_PATH = 1
    TURN_TO_OBJECT_OR_POINT = 2,
    DRIVE_TO_OPTIMAL_POSITION = 3,
    COLLECT_BALL = 4,
    DELIVER_BALL = 5

class Server:
    def __init__(self, fakeEv3Connection=False):
        self.host = '0.0.0.0'
        self.port = 12346

        self.SEND_CUSTOM_INSTRUCTIONS = True # Start instruction testing loop
        self.CONTROL_CUSTOM = False# Start custom control loop
        self.USE_PRE_MARKED_WALL = False # Launch window to mark walls on the course and use those.

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

        self.ai_model = AIModel("ball_detect/v10_t/weights/best.pt")
        self.course = Course()
        self.course_visualizer = CourseVisualizer(draw_boxes=True, draw_labels=True, draw_confidence=True, draw_masks=False, draw_walls=True, draw_direction_markers=True)
        self.path_planner = PathPlanner(strategy=AStarStrategyOptimized(obj_radius=config.LARGE_OBJECT_RADIUS))
        self.path_planner_visualizer = PathPlannerVisualizer()

        # extra 
        self.pure_pursuit_navigator = PurePursuitNavigator(None, 
                                                lookahead_distance=25, 
                                                max_speed=25, 
                                                true_max_speed=25, 
                                                kp=0.75, 
                                                max_turn_slowdown=1)
        
        self.pure_pursuit_navigator_slow = PurePursuitNavigator(None,
                                                lookahead_distance=10, 
                                                max_speed=5, 
                                                true_max_speed=5, 
                                                kp=0.75, 
                                                max_turn_slowdown=1)

        self.robot_state = RobotState.IDLE
        self.robot = None           # For storing previous robot if needed
        self.robot_direction = 0    # For storing previous robot direction if needed

        #####################
        self._key_map = {
            'e': ( 24,  24),
            's': (-24,  24),
            'd': (-24, -24),
            'f': ( 24, -24),
            'r': (  0,   0),
            'o': (  0,   0),  # open claw
            'p': (  0,   0),  # close claw
        }
        self._active_key = None
        # hook events
        if self.CONTROL_CUSTOM:
            for k in self._key_map:
                keyboard.on_press_key(k,   lambda e, k=k: self._activate(k))
                keyboard.on_release_key(k, lambda e, k=k: self._deactivate(k))

        #####################

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
    
    ###########
    def _activate(self, key):
        print(f"[SERVER] Activated key: {key}")
        self._active_key = key

    def _deactivate(self, key):
        print(f"[SERVER] Deactivated key: {key}")
        if self._active_key == key:
            self._active_key = None

    def control_custom_loop(self):
        stop_instr = {"cmd":"drive","left_speed":0,"right_speed":0}
        instr = stop_instr

        while True:
            ret, frame = self.cap.read()
            if not ret:
                continue

            self.course = self.ai_model.generate_course(frame)
            vis = self.course_visualizer.draw(frame, self.course)

            if keyboard.is_pressed('q'):
                break

            if self._active_key:
                if self._active_key == 'o':
                    instr = {"cmd":"claw","action":"open","speed":5}
                elif self._active_key == 'p':
                    instr = {"cmd":"claw","action":"close","speed":5}
                else:
                    ls, rs = self._key_map[self._active_key]
                    instr = {"cmd":"drive","left_speed":ls,"right_speed":rs}
            else:
                instr = stop_instr

            self.send_instruction(instr)
            cv2.imshow("view", vis)
            cv2.waitKey(1)

    '''
    def control_custom_loop(self):
        last_instruction =  {"cmd": "drive", "left_speed": 0, "right_speed": 0}

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
            else:
                last_instruction = {"cmd": "drive", "left_speed": 0, "right_speed": 0}
                # send the last instruction
                self.send_instruction(last_instruction)


            # show empty frame
            cv2.imshow("view",  current_video_frame_with_objs)
    '''

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

    def get_clicked_ball(self, x, y):
        # returns the ball that is clicked on, or None if no ball is clicked
        all_balls = self.course.get_white_balls() + self.course.get_orange_balls()
        for ball in all_balls:
            if self.course._bbox_within_threshold_point(ball.bbox, (x, y), threshold=0):
                print(f"[SERVER] Found ball at {ball.center}, clicked at ({x}, {y})")
                return ball
            
        print(f"[SERVER] No ball found at ({x}, {y})")
        return None
    
    def get_clicked_goal(self, x, y):
        # returns the goal that is clicked on, or None if no goal is clicked
        all_goals = self.course.get_goals()
        for goal in all_goals:
            if self.course._bbox_within_threshold_point(goal.bbox, (x, y), threshold=0):
                print(f"[SERVER] Found goal at {goal.center}, clicked at ({x}, {y})")
                return goal
            
        print(f"[SERVER] No goal found at ({x}, {y})")
        return None

    def custom_instruction_loop(self):
        current_state = RobotState.IDLE
        robot = None
        robot_direction = 0
        angle_to_target = -1 # used when in RobotState.TURN_TO_OBJECT
        spot = None
        is_edge_ball = False  # used to determine if the clicked ball is an edge ball

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
                self.pure_pursuit_navigator.set_path(None)
                instruction = {"cmd": "drive", "left_speed": 0, "right_speed": 0}
                self.send_instruction(instruction)
            elif key == ord('o'):
                # open claw
                instruction = {"cmd": "claw", "action": "open", "speed": 5}
                self.send_instruction(instruction)
            elif key == ord('p'):
                instruction = {"cmd": "claw", "action": "close", "speed": 5}
                self.send_instruction(instruction)
            # if key is number 1
            elif key == ord('d'):
                last_instruction = {"cmd": "deliver", "speed": 75}
                print(last_instruction)
                self.send_instruction(last_instruction)
            elif key == ord('1'):
                current_state = RobotState.FOLLOW_PATH
                self.pure_pursuit_navigator.set_path(None)
                instruction = {"cmd": "drive", "left_speed": 0, "right_speed": 0}
                self.send_instruction(instruction)
            elif key == ord('2'):
                current_state = RobotState.TURN_TO_OBJECT_OR_POINT
                self.pure_pursuit_navigator.set_path(None)
                instruction = {"cmd": "drive", "left_speed": 0, "right_speed": 0}
                self.send_instruction(instruction)
            elif key == ord('3'):
                current_state = RobotState.DRIVE_TO_OPTIMAL_POSITION
                self.pure_pursuit_navigator.set_path(None)
                instruction = {"cmd": "drive", "left_speed": 0, "right_speed": 0}
                self.send_instruction(instruction)
            elif key == ord('4'):
                current_state = RobotState.COLLECT_BALL
                self.pure_pursuit_navigator.set_path(None)
                instruction = {"cmd": "drive", "left_speed": 0, "right_speed": 0}
                self.send_instruction(instruction)
            elif key == ord('5'):
                current_state = RobotState.DELIVER_BALL
                self.pure_pursuit_navigator.set_path(None)
                instruction = {"cmd": "drive", "left_speed": 0, "right_speed": 0}
                self.send_instruction(instruction)

            # Handle mouse click to generate path
            if self.mouse_clicked_coords[0] is not None:
                x, y = self.mouse_clicked_coords[0]
                self.mouse_clicked_coords[0] = None

                if current_state == RobotState.FOLLOW_PATH:
                    grid = self.path_planner.generate_grid(self.course) # change to True if you want to drw floor
                    grid_img = self.path_planner_visualizer.draw_grid_objects(grid)
                    start = robot.center
                    start = (int(start[0]), int(start[1]))
                    end = (int(x), int(y))
                    print(f"[SERVER] Generating path from {start} to {end}...")
                    current_path = self.path_planner.find_path(start, end, grid)
                    if current_path is None or len(current_path) == 0:
                        print("[SERVER] No path found, please try again.")
                        self.pure_pursuit_navigator.set_path(None)
                        continue
                    self.pure_pursuit_navigator.set_path(current_path)
                    print(f"[SERVER] Path found: {len(current_path)} points.")
                    cv2.imshow("grid_visualization", grid_img)
                elif current_state == RobotState.TURN_TO_OBJECT_OR_POINT:
                    #closest_obj = self.course.get_nearest_object((x, y))
                    #if closest_obj is not None:
                        dst_point = (x, y)
                        src_point = robot.center
                        print(f"[SERVER] Turning to object {(x,y)} at {dst_point} from {src_point}...")
                        angle_to_target = angle_to(src_point, dst_point)
                elif current_state == RobotState.DRIVE_TO_OPTIMAL_POSITION:
                    # Find the optimal position to drive to
                    all_balls = self.course.get_white_balls() + self.course.get_orange_balls()
                    clicked_ball = None
                    for ball in all_balls:
                        if self.course._bbox_within_threshold_point(ball.bbox, (x, y), threshold=0):
                            print(f"[SERVER] Found ball at {spot}, driving to optimal position...")
                            clicked_ball = ball
                            break

                    if clicked_ball is not None:
                        # Calculate the optimal position to drive to
                        optimal_position = self.course.get_optimal_ball_parking_spot(clicked_ball)
                        if optimal_position is not None:
                            print(f"[SERVER] Optimal position to drive to: {optimal_position}")
                            spot = optimal_position
                            current_path = self.path_planner.find_path(robot.center, optimal_position, self.path_planner.generate_grid(self.course, False))
                            if current_path is not None and len(current_path) > 0:
                                self.pure_pursuit_navigator.set_path(current_path)
                                print(f"[SERVER] Path found: {len(current_path)} points.")
                            else:
                                print("[SERVER] No path found to optimal position.")
                        else:
                            print("[SERVER] Could not find an optimal position to drive to.")
                elif current_state == RobotState.COLLECT_BALL:
                    # Find the clicked ball (assuming your facing toward it)
                    clicked_ball = self.get_clicked_ball(x, y)
                    if clicked_ball is not None:
                        print(f"[SERVER] Collecting ball at {clicked_ball.center}...")
                        # Set the path to the clicked ball
                        is_edge_ball = self.course.is_ball_near_wall(clicked_ball)
                        self.path_planner.set_object_radius(config.SMALL_OBJECT_RADIUS)
                        current_path = self.path_planner.find_path(robot.center, clicked_ball.center, self.path_planner.generate_grid(self.course, excluded_objects=[clicked_ball]))
                        if current_path is not None and len(current_path) > 0:
                            self.pure_pursuit_navigator_slow.set_path(current_path)
                            print(f"[SERVER] Path found: {len(current_path)} points.")
                        else:
                            print("[SERVER] No path found to collect the ball.")
                    else:
                        print("[SERVER] No ball found at the clicked position.")
                elif current_state == RobotState.DELIVER_BALL:
                    # Find the clicked goal
                    #clicked_goal = self.get_clicked_goal(x, y)
                    spot = (x, y)
                    excluded_ball = self.course.get_nearest_ball(self.course.get_robot().center)
                    current_path = self.path_planner.find_path(robot.center, (x,y), self.path_planner.generate_grid(self.course, excluded_objects=[excluded_ball]))
                    if current_path is not None and len(current_path) > 0:
                        self.pure_pursuit_navigator.set_path(current_path)
                        print(f"[SERVER] Path found: {len(current_path)} points.")
                    else:
                        print("[SERVER] No path found to deliver the ball.")



            # If currently following a path
            if (self.pure_pursuit_navigator.path is not None) and current_state == RobotState.FOLLOW_PATH:
                if len(self.pure_pursuit_navigator.path) == 0:
                    print("[SERVER] No path to follow, please generate a path first.")
                    continue
                if spot is not None:
                    self.course_visualizer.highlight_point(current_video_frame_with_objs, spot, color=(0, 255, 0), radius=10)
                current_video_frame_with_objs = self.path_planner_visualizer.draw_path(current_video_frame_with_objs, current_path)
                instruction = self.pure_pursuit_navigator.compute_drive_command(robot.center, robot_direction)
                self.send_instruction(instruction)
                if distance(robot.center, current_path[-1]) < 10:
                    print("[SERVER] Reached the end of the path.")
                    self.pure_pursuit_navigator.set_path(None)
                    instruction = {"cmd": "drive", "left_speed": 0, "right_speed": 0}
                    self.send_instruction(instruction)
            elif current_state == RobotState.TURN_TO_OBJECT_OR_POINT and angle_to_target != -1:
                # Compute turn command to face the target object
                instruction = self.pure_pursuit_navigator.compute_turn_command(robot_direction, angle_to_target, newKp=0.9, new_max_speed=10)
                self.send_instruction(instruction)

                # Check if the robot is facing the target object
                print(f"[SERVER] Robot direction: {robot_direction}, angle to target: {angle_to_target}")
                print(f"[SERVER] Angle difference: {abs(angle_to_target - robot_direction)}")
                if abs(angle_to_target - robot_direction) < 3:
                    print("[SERVER] Robot is now facing the target object.")
                    angle_to_target = -1  # Reset angle to target
                    self.send_instruction({"cmd": "drive", "left_speed": 0, "right_speed": 0})
            elif (self.pure_pursuit_navigator.path is not None) and current_state == RobotState.DRIVE_TO_OPTIMAL_POSITION:
                if len(self.pure_pursuit_navigator.path) == 0:
                    print("[SERVER] No path to follow, please generate a path first.")
                    continue
                if spot is not None:
                    self.course_visualizer.highlight_point(current_video_frame_with_objs, spot, color=(0, 255, 0), radius=10)
                current_video_frame_with_objs = self.path_planner_visualizer.draw_path(current_video_frame_with_objs, current_path)
                instruction = self.pure_pursuit_navigator.compute_drive_command(robot.center, robot_direction)
                self.send_instruction(instruction)
                if distance(robot.center, current_path[-1]) < 10:
                    print("[SERVER] Reached the optimal position.")
                    self.pure_pursuit_navigator.set_path(None)
                    instruction = {"cmd": "drive", "left_speed": 0, "right_speed": 0}
                    self.send_instruction(instruction)
            elif (self.pure_pursuit_navigator_slow.path is not None) and current_state == RobotState.COLLECT_BALL:
                if len(self.pure_pursuit_navigator_slow.path) == 0:
                    print("[SERVER] No path to follow, please generate a path first.")
                    continue
                current_video_frame_with_objs = self.path_planner_visualizer.draw_path(current_video_frame_with_objs, current_path)
                instruction = self.pure_pursuit_navigator_slow.compute_drive_command(robot.center, robot_direction)
                self.send_instruction(instruction)
                stop_dist = 10 if is_edge_ball else 32  # Stop distance for edge balls is smaller
                print("Is edge ball:", is_edge_ball)
                if distance(robot.center, current_path[-1]) < stop_dist:
                    print("[SERVER] Reached the end of the path.")
                    instruction = {"cmd": "claw", "action": "close"}
                    self.send_instruction(instruction)
                    self.pure_pursuit_navigator_slow.set_path(None)
                    instruction = {"cmd": "drive", "left_speed": 0, "right_speed": 0}
                    self.send_instruction(instruction)
                    self.path_planner.set_object_radius(config.LARGE_OBJECT_RADIUS)  # Reset to large object radius for next path
                    is_edge_ball = False  # Reset edge ball flag
            elif (self.pure_pursuit_navigator.path is not None) and current_state == RobotState.DELIVER_BALL:
                if len(self.pure_pursuit_navigator.path) == 0:
                    print("[SERVER] No path to follow, please generate a path first.")
                    continue
                if spot is not None:
                    self.course_visualizer.highlight_point(current_video_frame_with_objs, spot, color=(0, 255, 0), radius=10)
                current_video_frame_with_objs = self.path_planner_visualizer.draw_path(current_video_frame_with_objs, current_path)
                instruction = self.pure_pursuit_navigator.compute_drive_command(robot.center, robot_direction)
                self.send_instruction(instruction)
                if distance(robot.center, current_path[-1]) < 25:
                    print("[SERVER] Reached the end of the path.")
                    self.pure_pursuit_navigator.set_path(None)
                    instruction = {"cmd": "drive", "left_speed": 0, "right_speed": 0}
                    self.send_instruction(instruction)
            


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