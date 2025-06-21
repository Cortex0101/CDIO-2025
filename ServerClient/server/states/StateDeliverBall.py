from .StateBase import StateBase
import config
import logging

import cv2

import time

from Course import Course, CourseObject

logger = logging.getLogger(__name__)


class StateDeliverBall(StateBase):
    def __init__(self, server, target_object=None): # might not need target_object as its just the closest ball to the robot that should be ignored
        super().__init__(server)
        self.server = server  # Reference to the main Server object
        self.target_object = target_object  # This can be used to specify a specific ball to deliver
        logger.debug("Initialized StateDeliverBall with target_object: %s", target_object)

    def on_enter(self):
        if self.target_object is None:
            # fire delivery method:
            instruction = {"cmd": "deliver", "speed": 75, "to_pos": 45}  # Open to 55 degrees
            self.server.send_instruction(instruction)
            time.sleep(3)  # wait for the delivery to complete
            
            # go to idle
            from .StateIdle import StateIdle
            self.server.set_state(StateIdle(self.server))
            return
        
        logger.debug("Entering StateDeliverBall with target_object: %s", self.target_object)
        self.GOAL_LOCATION = self.target_object.center
        robot = super().get_last_valid_robot() # will return a valid robot, or go to idle state if not found
        self.robot_center = robot.center
        self.robot_direction = robot.direction
        
        logger.debug("Trying to find a path to deliver the ball.")
        self.target_location = self.server.course.get_optimal_goal_parking_spot(self.GOAL_LOCATION)
        logger.debug(f"Target location for ball delivery: {self.target_location}")
        excluded_ball = self.server.course.get_nearest_ball(self.robot_center, color='either')
        self.grid = self.server.path_planner.generate_grid(self.server.course, excluded_objects=[excluded_ball] if excluded_ball is not None else [])

        self.current_path = self.server.path_planner.find_path(self.robot_center, self.target_location, self.grid)
        if self.current_path is not None and len(self.current_path) > 0:
            self.server.pure_pursuit_navigator.set_path(self.current_path)
            logger.info(f"[SERVER] Path found to deliver the ball: {len(self.current_path)} points.")
        else:
            logger.error("[SERVER] No path found to deliver the ball.")
            # If no path is found, go back to idle state
            instruction = {"cmd": "drive", "left_speed": 0, "right_speed": 0}
            self.server.send_instruction(instruction)
            self.server.pure_pursuit_navigator.set_path(None)
            from .StateIdle import StateIdle
            self.server.set_state(StateIdle(self.server))

            #logger.error("[SERVER] No path found to deliver the ball.")
            #from .StateIdle import StateIdle
            #self.server.set_state(StateIdle(self.server))

    def update(self, frame):
        """
        Update the state with the current frame.
        This method is called periodically to update the state.
        """
        robot = super().get_last_valid_robot() # will return a valid robot, or go to idle state if not found
        self.robot_center = robot.center
        self.robot_direction = robot.direction

        frame = self.server.path_planner_visualizer.draw_path(frame, self.server.pure_pursuit_navigator.path)
        self.server.course_visualizer.highlight_point(frame, self.target_location, color=(255, 255, 0))
        instruction = self.server.pure_pursuit_navigator.compute_drive_command(self.robot_center, self.robot_direction)
        self.server.send_instruction(instruction)
        
        if self._distance(self.robot_center, self.server.pure_pursuit_navigator.path[-1]) < config.REACHED_POINT_DISTANCE:
            logger.info("[SERVER] Reached the end of the path.")
            self.server.pure_pursuit_navigator.set_path(None)
            instruction = {"cmd": "drive", "left_speed": 0, "right_speed": 0}
            self.server.send_instruction(instruction)
            from .StateRotateToObject import StateRotateToObject
            logger.debug("Switching to RotateToObject state to face the goal")
            self.server.set_state(StateRotateToObject(self.server, target_object=self.target_object))

        return frame


    def on_exit(self):
        pass

    def attempt_to_unstuck(self, frame):
        logger.info("Trying to unstuck: backing up and retrying.")
        # Example: send a reverse command, or transition to a recovery state
        instruction = {"cmd": "drive_seconds", "seconds": 2, "speed": -10}
        self.server.send_instruction(instruction)
        time.sleep(2)
        return frame

    def on_click(self, event, x, y):
        """
        Handle click events. Default implementation does nothing.
        Override in subclasses if needed.
        """
        pass

    def on_key_press(self, key):
        if key == ord('x'):
            # If 'g' is pressed, go back to idle state
            from .StateIdle import StateIdle
            self.server.set_state(StateIdle(self.server))

    def _distance(self, a, b):
        """
        Calculate the Euclidean distance between two points.
        """
        d = ((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) ** 0.5
        logger.debug(f"Calculated distance between {a} and {b}: {d}")
        return d

'''
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


'''