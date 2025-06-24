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
#import keyboard

import cProfile
import pstats

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

class Server:
    def __init__(self):
        self.host = '0.0.0.0'
        self.port = 12346

        self.SEND_CUSTOM_INSTRUCTIONS = False # Start instruction testing loop
        self.CONTROL_CUSTOM = False# Start custom control loop

        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            logger.critical("Could not open camera, exiting.")
            exit()

        self.mouse_clicked_coords = [None]
        cv2.namedWindow("view")
        cv2.setMouseCallback("view", self._on_click)

        self.ai_model = AIModel(model_path=config.YOLO_MODEL_15_L, min_confidence=config.YOLO_MODEL_MINIMUM_CONFIDENCE)
        self.course = Course()
        self.course_visualizer = CourseVisualizer(draw_boxes=True, draw_labels=True, draw_confidence=True, draw_masks=False, draw_walls=True, draw_direction_markers=True)
        self.path_planner = PathPlanner(strategy=AStarStrategyOptimized(obj_radius=config.LARGE_OBJECT_RADIUS))
        self.path_planner_visualizer = PathPlannerVisualizer()

        self.last_valid_robot = None  # For storing the last valid robot position
        self.last_valid_large_goal = None

        # extra 
        self.pure_pursuit_navigator = PurePursuitNavigator(None, 
                                                lookahead_distance=config.FAST_LOOKAHEAD_DISTANCE,
                                                max_speed=config.FAST_MAX_SPEED,
                                                true_max_speed=config.FAST_MAX_SPEED,
                                                kp=config.FAST_KP, 
                                                max_turn_slowdown=config.FAST_MAX_TURN_SLOW)
        
        self.pure_pursuit_navigator_slow = PurePursuitNavigator(None,
                                                lookahead_distance=config.SLOW_LOOKAHEAD_DISTANCE,
                                                max_speed=config.SLOW_MAX_SPEED,
                                                true_max_speed=config.SLOW_MAX_SPEED, 
                                                kp=config.SLOW_KP,
                                                max_turn_slowdown=config.SLOW_MAX_TURN_SLOW)

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
                #keyboard.on_press_key(k,   lambda e, k=k: self._activate(k))
                #keyboard.on_release_key(k, lambda e, k=k: self._deactivate(k))
                pass
        
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
                if self.CONTROL_CUSTOM:
                    self.control_custom_loop()
                else:
                    self.main_loop()
            except Exception:
                logger.error(traceback.format_exc())
            finally:
                try:
                    self.conn.close()
                except Exception:
                    logger.error("Error closing connection: %s", traceback.format_exc())
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

            #if keyboard.is_pressed('q'):
             #   break

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
            logger.error(traceback.format_exc())
            return {"status": "error", "msg": str(e)}

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
        self.set_state(StateIdle(self))

        self.prev_time = time.time()

        while True:
            frame = self._capture_frame()
            if frame is None: continue

            self.course = self.ai_model.generate_course(frame)
            #frame = self.course_visualizer.draw(frame, self.course)
            frame = self.ai_model.results[0].plot()

            self._on_key_press() # send key press to current state
            try:
                frame = self.current_state.step(frame)  # update current state
            except Exception as e:
                # log error with traceback
                logger.error(traceback.format_exc())

            self.display_fps(frame)

            cv2.imshow("view", frame)

def main():
    configure_logging()
    server = Server()  # Set to True for fake EV3 connection

if __name__ == "__main__":
    main()