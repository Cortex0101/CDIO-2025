from .StateBase import StateBase
import config

import logging
logger = logging.getLogger(__name__)

import cv2

'''
 CLAW_OPEN_POS = 80
    CLAW_CLOSED_POS = 0
    CLAW_DELIVER_POS = 50  # position to deliver the ball
'''

class StateCalibration(StateBase):
    ORIGINAL_PARAMETERS = {
        "fast_pure_pursuit_navigator": {
            "Kp": config.FAST_KP,
            "max_speed": config.FAST_MAX_SPEED,
            "lookahead_distance": config.FAST_LOOKAHEAD_DISTANCE
        },
        "slow_pure_pursuit_navigator": {
            "Kp": config.SLOW_KP,
            "max_speed": config.SLOW_MAX_SPEED,
            "lookahead_distance": config.SLOW_LOOKAHEAD_DISTANCE
        },
        "robot": {
            "claw_closed_position": 0,  # position to close the claw
            "claw_open_position": 80,     # position to open the claw
            "claw_deliver_position": 50   # position to open the claw when collecting the ball
        }
    }

    def __init__(self, server):
        super().__init__(server)
        logger.debug("Initialized StateCalibration.")

        self.parameters = {
            "fast_pure_pursuit_navigator": {
                "Kp": config.FAST_KP,
                "max_speed": config.FAST_MAX_SPEED,
                "lookahead_distance": config.FAST_LOOKAHEAD_DISTANCE,
                "max_turn_slow": config.FAST_MAX_TURN_SLOW
            },
            "slow_pure_pursuit_navigator": {
                "Kp": config.SLOW_KP,
                "max_speed": config.SLOW_MAX_SPEED,
                "lookahead_distance": config.SLOW_LOOKAHEAD_DISTANCE,
                "max_turn_slow": config.SLOW_MAX_TURN_SLOW
            },
            "robot": {
                "claw_closed_position": 0,  # position to close the claw
                "claw_open_position": 80,     # position to open the claw
                "claw_deliver_position": 50   # position to open the claw when collecting the ball
            }
        }

        self.increment_by_1 = [
            "max_speed",
            "lookahead_distance",
            "claw_closed_position",
            "claw_open_position",
            "claw_deliver_position"
        ]

        self.increment_by_0point1 = [
            "Kp",
            "max_turn_slow"
        ]

        self.use_slow_navigator = False  # Flag to switch between fast and slow navigator

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

        if self.server.pure_pursuit_navigator.path is None:
            return self._scale_frame(frame)
        if len(self.server.pure_pursuit_navigator.path) == 0:
            return self._scale_frame(frame)
        
        if self.server.course.get_robot() is not None:
            robot = self.server.course.get_robot()
            self.robot_center = robot.center
            self.robot_direction = robot.direction
        else:
            logger.warning("No robot found in the course, using previous position.")

        frame = self.server.path_planner_visualizer.draw_path(frame, self.server.pure_pursuit_navigator.path)
        instruction = None
        if self.use_slow_navigator:
            instruction = self.server.pure_pursuit_navigator_slow.compute_drive_command(self.robot_center, self.robot_direction)
        else:
            instruction = self.server.pure_pursuit_navigator.compute_drive_command(self.robot_center, self.robot_direction)
        self.server.send_instruction(instruction)

        if self._distance(self.robot_center, self.server.pure_pursuit_navigator.path[-1]) < config.REACHED_POINT_DISTANCE:
            logger.info("[SERVER] Reached the end of the path in calibration state.")
            self.server.pure_pursuit_navigator.set_path(None)
            self.server.pure_pursuit_navigator_slow.set_path(None)
            instruction = {"cmd": "drive", "left_speed": 0, "right_speed": 0}
            self.server.send_instruction(instruction)

        return self._scale_frame(frame)
    
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
            # scale back down
            x = int(x / self.scale)
            y = int(y / self.scale)
            self._setup_go_to_point((x, y))

    def on_key_press(self, key):
        """
        Handle key press events. Default implementation does nothing.
        Override in subclasses if needed.
        """
        if key == 0:
            return

        # User can navigate through parameters using 'n' and 'p' keys
        if key == ord('k'):
            self._next_parameter()
            
        elif key == ord('j'):
            self._previous_parameter()
            
        # Change the value of the current parameter with 'up' and 'down' keys
        elif key == ord('m'):
            # Increase the value of the current parameter
            param_name = self.current_parameter[1]
            if param_name in self.parameters[self.current_parameter[0]]:
                if param_name in self.increment_by_1:
                    self.parameters[self.current_parameter[0]][param_name] += 1
                elif param_name in self.increment_by_0point1:
                    self.parameters[self.current_parameter[0]][param_name] += 0.1
            
        elif key == ord('n'):
            # Decrease the value of the current parameter
            param_name = self.current_parameter[1]
            if param_name in self.parameters[self.current_parameter[0]]:
                if param_name in self.increment_by_1:
                    self.parameters[self.current_parameter[0]][param_name] -= 1
                elif param_name in self.increment_by_0point1:
                    self.parameters[self.current_parameter[0]][param_name] -= 0.1
        elif key == ord('s'):
            self.use_slow_navigator = not self.use_slow_navigator
        elif key == ord('r'):
            self.parameters = self.ORIGINAL_PARAMETERS.copy()
            self.current_parameter = ("fast_pure_pursuit_navigator", "Kp")
        elif key == ord('l'):
            self.server.pure_pursuit_navigator.kp = self.parameters["fast_pure_pursuit_navigator"]["Kp"]
            self.server.pure_pursuit_navigator.max_speed = self.parameters["fast_pure_pursuit_navigator"]["max_speed"]
            self.server.pure_pursuit_navigator.true_max_speed = self.parameters["fast_pure_pursuit_navigator"]["max_speed"]
            self.server.pure_pursuit_navigator.lookahead_distance = self.parameters["fast_pure_pursuit_navigator"]["lookahead_distance"]

            self.server.pure_pursuit_navigator_slow.kp = self.parameters["slow_pure_pursuit_navigator"]["Kp"]
            self.server.pure_pursuit_navigator_slow.max_speed = self.parameters["slow_pure_pursuit_navigator"]["max_speed"]
            self.server.pure_pursuit_navigator_slow.true_max_speed = self.parameters["slow_pure_pursuit_navigator"]["max_speed"]
            self.server.pure_pursuit_navigator_slow.lookahead_distance = self.parameters["slow_pure_pursuit_navigator"]["lookahead_distance"]

            instruction = {"cmd": "set_open_pos", "pos": self.parameters["robot"]["claw_open_position"]}
            self.server.send_instruction(instruction)
            instruction = {"cmd": "set_closed_pos", "pos": self.parameters["robot"]["claw_closed_position"]}
            self.server.send_instruction(instruction)
            instruction = {"cmd": "set_deliver_pos", "pos": self.parameters["robot"]["claw_deliver_position"]}
            self.server.send_instruction(instruction)
        else:
            logger.debug(f"Key pressed: {key}, no action defined for this key.")

    def _next_parameter(self):
        """
        Move to the next parameter in the calibration.
        If at the end, of parameters for one navigator, switch to the other navigator.

        Flow would be:
        fast_pure_pursuit_navigator, Kp
        fast_pure_pursuit_navigator, max_speed
        fast_pure_pursuit_navigator, lookahead_distance
        pure_pursuit_navigator_slow, Kp
        slow_pure_pursuit_navigator, max_speed
        slow_pure_pursuit_navigator, lookahead_distance 
        fast_pure_pursuit_navigator, Kp # wraps around
        """
        outer_keys = list(self.parameters.keys())
        current_outer_index = outer_keys.index(self.current_parameter[0])
        inner_keys = list(self.parameters[self.current_parameter[0]].keys())
        current_inner_index = inner_keys.index(self.current_parameter[1])

        # Move to the next parameter
        if current_inner_index < len(inner_keys) - 1:
            # Move to the next parameter in the same navigator
            next_inner_key = inner_keys[current_inner_index + 1]
            self.current_parameter = (self.current_parameter[0], next_inner_key)
        else:
            # Move to the first parameter of the next navigator
            if current_outer_index < len(outer_keys) - 1:
                next_outer_key = outer_keys[current_outer_index + 1]
                self.current_parameter = (next_outer_key, list(self.parameters[next_outer_key].keys())[0])
            else:
                # Wrap around to the first navigator
                self.current_parameter = (outer_keys[0], inner_keys[0])

    def _previous_parameter(self):
        """
        Move to the previous parameter in the calibration.
        If at the beginning of parameters for one navigator, switch to the other navigator.

        Flow would be:
        fast_pure_pursuit_navigator, Kp
        fast_pure_pursuit_navigator, max_speed
        fast_pure_pursuit_navigator, lookahead_distance
        pure_pursuit_navigator_slow, Kp
        slow_pure_pursuit_navigator, max_speed
        slow_pure_pursuit_navigator, lookahead_distance 
        fast_pure_pursuit_navigator, Kp # wraps around
        """
        outer_keys = list(self.parameters.keys())
        current_outer_index = outer_keys.index(self.current_parameter[0])
        inner_keys = list(self.parameters[self.current_parameter[0]].keys())
        current_inner_index = inner_keys.index(self.current_parameter[1])

        # Move to the previous parameter
        if current_inner_index > 0:
            # Move to the previous parameter in the same navigator
            next_inner_key = inner_keys[current_inner_index - 1]
            self.current_parameter = (self.current_parameter[0], next_inner_key)
        else:
            # Move to the last parameter of the previous navigator
            if current_outer_index > 0:
                next_outer_key = outer_keys[current_outer_index - 1]
                self.current_parameter = (next_outer_key, list(self.parameters[next_outer_key].keys())[-1])
            else:
                # Wrap around to the last navigator
                self.current_parameter = (outer_keys[-1], inner_keys[-1])

    def _scale_frame(self, frame):
        """
        Scale the frame to fit the window size.
        """
        height, width = frame.shape[:2]
        self.scale = min(1920 / width, 1080 / height)
        new_size = (int(width * self.scale), int(height * self.scale))
        scaled_frame = cv2.resize(frame, new_size, interpolation=cv2.INTER_LINEAR)
        return scaled_frame
        

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
        self.server.pure_pursuit_navigator_slow.set_path(path)
        return True

    