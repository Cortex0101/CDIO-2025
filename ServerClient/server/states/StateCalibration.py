from .StateBase import StateBase
import config

import logging
logger = logging.getLogger(__name__)

import cv2

class StateCalibration(StateBase):
    def __init__(self, server):
        super().__init__(server)
        logger.debug("Initialized StateCalibration.")

        self.parameters = {
            "fast_pure_pursuit_navigator": {
                "Kp": config.FAST_KP,
                "max_speed": config.FAST_MAX_SPEED,
                "lookahead_distance": config.FAST_LOOKAHEAD_DISTANCE
            },
            "slow_pure_pursuit_navigator": {
                "Kp": config.SLOW_KP,
                "max_speed": config.SLOW_MAX_SPEED,
                "lookahead_distance": config.SLOW_LOOKAHEAD_DISTANCE
            }
        }

        self.current_parameter = ("fast_pure_pursuit_navigator", "Kp")

    def on_enter(self):
        logger.info("Entering StateCalibration.")
        robot = super().get_last_valid_robot() # will return a valid robot, or go to idle state if not found
        if robot is not None:
            self.robot_center = robot.center
            self.robot_direction = robot.direction

    def update(self, frame):
        robot = super().get_last_valid_robot()  # will return a valid robot, or go to idle state if not found
        if robot is not None:
            self.robot_center = robot.center
            self.robot_direction = robot.direction

        # draw the current parameters on the frame
        cv2.putText(frame, f"Current Parameter: {self.current_parameter[0]} - {self.current_parameter[1]}: {self.parameters[self.current_parameter[0]][self.current_parameter[1]]}",
                    (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        if len(self.server.pure_pursuit_navigator.path) == 0:
            return frame
        
        if self.server.course.get_robot() is not None:
            robot = self.server.course.get_robot()
            self.robot_center = robot.center
            self.robot_direction = robot.direction
        else:
            logger.warning("No robot found in the course, using previous position.")

        frame = self.server.path_planner_visualizer.draw_path(frame, self.server.pure_pursuit_navigator.path)
        instruction = self.server.pure_pursuit_navigator.compute_drive_command(self.robot_center, self.robot_direction)
        self.server.send_instruction(instruction)

        if self._distance(self.robot_center, self.server.pure_pursuit_navigator.path[-1]) < config.REACHED_POINT_DISTANCE:
            logger.info("[SERVER] Reached the end of the path in calibration state.")
            self.server.pure_pursuit_navigator.set_path(None)
            instruction = {"cmd": "drive", "left_speed": 0, "right_speed": 0}
            self.server.send_instruction(instruction)
    
    def attempt_to_unstuck(self, frame):
        return self.update(frame)  # No specific unstuck logic implemented, just update the state

    def on_exit(self):
        logger.info("Exiting StateCalibration.")

    def on_click(self, event, x, y):
        """
        Handle click events. Default implementation does nothing.
        Override in subclasses if needed.
        """
        # if left click
        if event == cv2.EVENT_LBUTTONDOWN:
            logger.info(f"Clicked at ({x}, {y}) in calibration state")
            self._setup_go_to_point((x, y))

    def on_key_press(self, key):
        """
        Handle key press events. Default implementation does nothing.
        Override in subclasses if needed.
        """
        # User can navigate through parameters using 'n' and 'p' keys
        if key == ord('n'):
            # Move to the next parameter
            keys = list(self.parameters.keys())
            current_index = keys.index(self.current_parameter[0])
            next_index = (current_index + 1) % len(keys)
            self.current_parameter = (keys[next_index], self.current_parameter[1])
            logger.info(f"Switched to next parameter: {self.current_parameter}")
        elif key == ord('p'):
            # Move to the previous parameter
            keys = list(self.parameters.keys())
            current_index = keys.index(self.current_parameter[0])
            prev_index = (current_index - 1) % len(keys)
            self.current_parameter = (keys[prev_index], self.current_parameter[1])
            logger.info(f"Switched to previous parameter: {self.current_parameter}")
        # Change the value of the current parameter with 'up' and 'down' keys
        elif key == ord('u'):
            # Increase the value of the current parameter
            if self.current_parameter[1] in self.parameters[self.current_parameter[0]]:
                self.parameters[self.current_parameter[0]][self.current_parameter[1]] += 0.1
                logger.info(f"Updated {self.current_parameter[0]} - {self.current_parameter[1]} to {self.parameters[self.current_parameter[0]][self.current_parameter[1]]}")
        elif key == ord('d'):
            # Decrease the value of the current parameter
            if self.current_parameter[1] in self.parameters[self.current_parameter[0]]:
                self.parameters[self.current_parameter[0]][self.current_parameter[1]] -= 0.1
                logger.info(f"Updated {self.current_parameter[0]} - {self.current_parameter[1]} to {self.parameters[self.current_parameter[0]][self.current_parameter[1]]}")
        
        '''
        PurePursuitNavigator(None, 
                                                lookahead_distance=config.FAST_LOOKAHEAD_DISTANCE,
                                                max_speed=config.FAST_MAX_SPEED,
                                                true_max_speed=config.FAST_MAX_SPEED,
                                                kp=config.FAST_KP, 
                                                max_turn_slowdown=1)
        '''
        self.server.pure_pursuit_navigator.kp = self.parameters["fast_pure_pursuit_navigator"]["Kp"]
        self.server.pure_pursuit_navigator.max_speed = self.parameters["fast_pure_pursuit_navigator"]["max_speed"]
        self.server.pure_pursuit_navigator.lookahead_distance = self.parameters["fast_pure_pursuit_navigator"]["lookahead_distance"]

    def _distance(self, a, b):
        """
        Calculate the Euclidean distance between two points.
        """
        d = ((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) ** 0.5
        logger.debug(f"Calculated distance between {a} and {b}: {d}")
        return d

    def _setup_go_to_point(self, point):
        grid = self.server.path_planner.generate_grid(self.server.course)
        path = self.server.path_planner.find_path(self.robot_center, point, grid)
        if path is None or len(path) == 0:
            logger.warning(f"No path found to ball point: {point}. Try clicking again to set a new point.")
            return False
        self.server.pure_pursuit_navigator.set_path(path)
        return True

    