import socket
import json
import time
import math
import sys
import select
import threading
import queue
import traceback
from enum import Enum

import cv2
import keyboard

from AImodel import *
from Course import *
from PathPlanner import *
from PurePursuit import *
import config

from states.StateIdle import StateIdle
from states.StateGoToNearestBall import StateGoToNearestBall
from states.StateCollectBall import StateCollectBall
from states.StateRotateToObject import StateRotateToObject
from states.StateCalibration import StateCalibration

import logging
from logging.handlers import RotatingFileHandler

def configure_logging():
    # Root logger configuration: log INFO+ to console, DEBUG+ to file
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s():%(lineno)s - %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    # 1) Console handler
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(fmt, datefmt))

    # 2) Rotating file handler
    file_handler = RotatingFileHandler(
        "robot.log", maxBytes=5*1024*1024, backupCount=3
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(fmt, datefmt))

    # 3) Root logger
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(console)
    root.addHandler(file_handler)

logger = logging.getLogger(__name__)

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
    DELIVER_BALL = 5,
    SELECT_GOAL_CENTERS = 6 # for debugging at home

class Server:
    def __init__(self):
        self.host = '0.0.0.0'
        self.port = 12346

        self.SEND_CUSTOM_INSTRUCTIONS = False # Start instruction testing loop
        self.CONTROL_CUSTOM = False# Start custom control loop
        self.USE_PRE_MARKED_WALL = False # Launch window to mark walls on the course and use those.

        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            logger.critical("Could not open camera, exiting.")
            exit()

        self.mouse_clicked_coords = [None]
        cv2.namedWindow("view")
        cv2.setMouseCallback("view", self._on_click)
        cv2.namedWindow("grid_visualization")

        self.ai_model = AIModel(model_path=config.YOLO_MODEL_15_L, min_confidence=config.YOLO_MODEL_MINIMUM_CONFIDENCE)
        self.course = Course()
        self.course_visualizer = CourseVisualizer(draw_boxes=True, draw_labels=True, draw_confidence=True, draw_masks=False, draw_walls=True, draw_direction_markers=True)
        self.path_planner = PathPlanner(strategy=AStarStrategyOptimized(obj_radius=config.LARGE_OBJECT_RADIUS))
        self.path_planner_visualizer = PathPlannerVisualizer()

        self.last_valid_robot = None  # For storing the last valid robot position

        # extra 
        self.pure_pursuit_navigator = PurePursuitNavigator(None, 
                                                lookahead_distance=config.FAST_LOOKAHEAD_DISTANCE,
                                                max_speed=config.FAST_MAX_SPEED,
                                                true_max_speed=config.FAST_MAX_SPEED,
                                                kp=config.FAST_KP, 
                                                max_turn_slowdown=1)
        
        self.pure_pursuit_navigator_slow = PurePursuitNavigator(None,
                                                lookahead_distance=config.SLOW_LOOKAHEAD_DISTANCE,
                                                max_speed=config.SLOW_MAX_SPEED,
                                                true_max_speed=config.SLOW_MAX_SPEED, 
                                                kp=config.SLOW_KP,
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
        # fps variables calculations
        self.prev_time = time.time()
        self.avg_fps = 0.0
        self.frame_count = 0

        # if send_custom_instructions is true, we will be able to simply send instruction by entering
        # them in the console, otherwise we will have to use we will run the main_loop() method
        self.accept_connections_loop()

        cv2.destroyAllWindows()

    def accept_connections_loop(self):
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind((self.host, self.port))
        self.server.listen(1)
        logger.info(f"Server started on {self.host}:{self.port}, waiting for EV3 connection.")

        while True:
            self.conn, addr = self.server.accept()
            logger.info(f"Connected to EV3 at {addr}")
            try:
                if self.SEND_CUSTOM_INSTRUCTIONS:
                    self.custom_instruction_loop()
                elif self.CONTROL_CUSTOM:
                    self.control_custom_loop()
                else:
                    self.main_loop()
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                # more info
                exc_type, exc_value, exc_traceback = sys.exc_info()
                logger.error(f"Exception type: {exc_type}")
                logger.error(f"Exception value: {exc_value}")
                logger.error("Traceback:")
                traceback.print_exception(exc_type, exc_value, exc_traceback)
            finally:
                try:
                    self.conn.close()
                except Exception:
                    pass
                logger.info("Connection closed, waiting for new EV3 connection.")

    def _activate(self, key):
        logger.info(f"Activated key: {key}")
        self._active_key = key

    def _deactivate(self, key):
        logger.info(f"Deactivated key: {key}")
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

    def send_instruction(self, instruction, wait_for_response=False):
        try:
            self.conn.sendall((json.dumps(instruction) + '\n').encode())
            if wait_for_response:
                data = self.conn.recv(1024)
                response = json.loads(data.decode())
                return response
        except Exception as e:
            logger.error(f"Error sending instruction: {e}")
            return {"status": "error", "msg": str(e)}

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            logger.info(f"Mouse clicked at ({x}, {y})")
            self.mouse_clicked_coords[0] = (x, y)

    def get_clicked_ball(self, x, y):
        # returns the ball that is clicked on, or None if no ball is clicked
        all_balls = self.course.get_white_balls() + self.course.get_orange_balls()
        for ball in all_balls:
            if self.course._bbox_within_threshold_point(ball.bbox, (x, y), threshold=0):
                logger.info(f"Found ball at {ball.center}, clicked at ({x}, {y})")
                return ball
            
        logger.info(f"No ball found at ({x}, {y})")
        return None
    
    def get_clicked_goal(self, x, y):
        # returns the goal that is clicked on, or None if no goal is clicked
        all_goals = self.course.get_goals()
        for goal in all_goals:
            if self.course._bbox_within_threshold_point(goal.bbox, (x, y), threshold=0):
                logger.info(f"Found goal at {goal.center}, clicked at ({x}, {y})")
                return goal
            
        logger.info(f"No goal found at ({x}, {y})")
        return None

    def custom_instruction_loop(self):
        current_state = RobotState.IDLE
        robot = None
        robot_direction = 0
        angle_to_target = -1 # used when in RobotState.TURN_TO_OBJECT
        angle_has_been_correct_for_x_frame = 0 # used to determine if the robot is facing the target object
        spot = None
        is_edge_ball = False  # used to determine if the clicked ball is an edge ball

        # remove later
        clicked_goal_center = None

        while True:
            ret, current_video_frame = self.cap.read()

            if not ret:
                logger.error("Could not read frame from camera.")
                continue

            self.course = self.ai_model.generate_course(current_video_frame)
            current_video_frame_with_objs = self.course_visualizer.draw(current_video_frame, self.course)

            # remove later
            if clicked_goal_center is not None:
                # Highlight the clicked goal center
                self.course_visualizer.highlight_point(current_video_frame_with_objs, clicked_goal_center, color=(0, 255, 0), radius=10)

            # robot and direction
            if self.course.get_robot() is None:
                logger.warning("No robot detected in course, using previous position.")
            else:
                robot = self.course.get_robot()
                if robot.direction is None:
                    logger.warning("Robot direction is None, using previous value.")
                else:
                    robot_direction = robot.direction

            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                logger.info("Quitting server.")
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
            elif key == ord('6'):
                current_state = RobotState.SELECT_GOAL_CENTERS
                self.pure_pursuit_navigator.set_path(None)
                instruction = {"cmd": "drive", "left_speed": 0, "right_speed": 0}
                self.send_instruction(instruction)

            # Handle mouse click to generate path
            if self.mouse_clicked_coords[0] is not None:
                x, y = self.mouse_clicked_coords[0]
                self.mouse_clicked_coords[0] = None

                # remove late ##################
                if current_state == RobotState.SELECT_GOAL_CENTERS:
                    # For debugging at home, select goal centers
                    print(f"[SERVER] Selected goal center at ({x}, {y})")
                    clicked_goal_center = (x, y)
                ################################


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
                        discarded_points = optimal_position[2]
                        optimal_position = (optimal_position[0], optimal_position[1])  # Get the position without discarded points
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
                elif current_state == RobotState.DELIVER_BALL and clicked_goal_center is not None:
                    # Find the clicked goal
                    #clicked_goal = self.get_clicked_goal(x, y)
                    spot = self.course.get_optimal_goal_parking_spot(clicked_goal_center)
                    excluded_ball = self.course.get_nearest_ball(self.course.get_robot().center, 'either')
                    current_path = self.path_planner.find_path(robot.center, spot, self.path_planner.generate_grid(self.course, excluded_objects=[excluded_ball]))
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
                    angle_has_been_correct_for_x_frame += 1
                    if angle_has_been_correct_for_x_frame > 5:  # Wait for 5 frames to ensure it's stable
                        print("[SERVER] Angle has been stable for 5 frames, switching to next state.")
                        angle_to_target = -1  # Reset angle to target
                        angle_has_been_correct_for_x_frame = 0
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
                stop_dist = 10 if is_edge_ball else 20  # Stop distance for edge balls is smaller
                print("Is edge ball:", is_edge_ball)
                if distance(robot.center, current_path[-1]) < stop_dist:
                    print("[SERVER] Reached the end of the path.")
                    instruction = {"cmd": "claw", "action": "close"}
                    self.send_instruction(instruction)
                    self.pure_pursuit_navigator_slow.set_path(None)
                    instruction = {"cmd": "drive", "left_speed": 0, "right_speed": 0}
                    self.send_instruction(instruction)
                    time.sleep(0.5)  # Wait for claw to close
                    if is_edge_ball:
                        instruction = {"cmd": "drive_seconds", "seconds": 2, "speed": -10}
                        self.send_instruction(instruction)  # Back off a bit for edge balls
                        time.sleep(2)
                    self.path_planner.set_object_radius(config.LARGE_OBJECT_RADIUS)  # Reset to large object radius for next path
                    is_edge_ball = False  # Reset edge ball flag
            elif (self.pure_pursuit_navigator.path is not None) and current_state == RobotState.DELIVER_BALL:
                if len(self.pure_pursuit_navigator.path) == 0:
                    print("[SERVER] No path to follow, please generate a path first.")
                    continue
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

    def set_state(self, new_state):
        if hasattr(self, 'current_state') and self.current_state is not None:
            self.current_state.on_exit()
        self.current_state = new_state
        self.current_state.on_enter()

    def _capture_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            logger.error("Failed to capture frame from camera.")
            return None
        return frame

    def _quit(self):
        logger.info("Quitting server.")
        cv2.destroyAllWindows()
        sys.exit(0)

    def _on_key_press(self):
        key = cv2.waitKey(1) & 0xFF

        if key == ord('Q'):
            self._quit()

        if key == ord('D'):
            instruction = {"cmd": "deliver", "speed": 75}
            self.send_instruction(instruction)

        if key == ord('O'):
            instruction = {"cmd": "claw", "action": "open", "speed": 5}
            self.send_instruction(instruction)

        if key == ord('P'):
            instruction = {"cmd": "claw", "action": "close", "speed": 5}
            self.send_instruction(instruction)

        if key == ord('Ø'): # log avg fps
            logger.info(f"Average FPS: {self.avg_fps:.2f} over {self.frame_count} frames")

        self.current_state.on_key_press(key)

    def _on_click(self, event, x, y, flags, param):
        self.current_state.on_click(event, x, y)

    def display_fps(self, frame):
        '''Display FPS at the top right corner of the frame.'''
        if frame is None:
            return
        curr_time = time.time()
        fps = 1 / (curr_time - self.prev_time)
        self.prev_time = curr_time
        cv2.putText(frame, f"FPS: {fps:.2f}", (frame.shape[1] - 150, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        self.frame_count += 1
        self.avg_fps = (self.avg_fps * (self.frame_count - 1) + fps) / self.frame_count

    def main_loop(self):
        self.set_state(StateIdle(self))  # Initialize the state
        #self.set_state(StateCalibration(self))

        self.prev_time = time.time()

        while True:
            frame = self._capture_frame()
            if frame is None: continue

            self.course = self.ai_model.generate_course(frame)
            frame = self.course_visualizer.draw(frame, self.course)

            self._on_key_press() # send key press to current state
            try:
                frame = self.current_state.step(frame)  # update current state
            except Exception as e:
                # log error with traceback
                logger.error(f"Error in current state update: {e}")
                logger.error(traceback.format_exc())

            self.display_fps(frame)

            cv2.imshow("view", frame)

if __name__ == "__main__":
    configure_logging()
    server = Server()  # Set to True for fake EV3 connection