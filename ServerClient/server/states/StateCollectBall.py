from .StateBase import StateBase
import config

import time

from Course import Course, CourseObject

import logging
logger = logging.getLogger(__name__)

class StateCollectBall(StateBase):
    def __init__(self, server, target_object=None):
        super().__init__(server)
        logger.debug("Initialized StateCollectBall with target_object: %s", target_object)
        self.target_object = target_object

    def on_enter(self):
        robot = self.server.last_valid_robot  # Get the last valid robot state
        self.robot_center = robot.center
        self.robot_direction = robot.direction
        logger.debug(f"Check if target_object is near a wall: {self.target_object}")
        self.is_edge_ball = self.server.course.is_ball_near_wall(self.target_object)
        self.is_somewhat_near_cross = self.server.course.is_ball_near_cross(self.target_object, threshold=config.SOMEWHAT_NEAR_CROSS_THRESHOLD)

        ### quick fix, as it does not recognize the target object, passed from earlier ###
        if self.target_object is None:
            logger.warning("Target object is None, trying to find the nearest ball.")
            new_target = self.server.course.get_nearest_ball(self.robot_center, color='white')
            if new_target is not None:
                logger.info(f"Found nearest ball: {new_target}, using it as target object.")
                self.target_object = new_target
            else:
                logger.error("No nearest ball found, cannot proceed. Going to idle state.")
                from .StateIdle import StateIdle
                self.server.set_state(StateIdle(self.server))
                return
        ###################

        if self.is_edge_ball:
            self.server.path_planner.set_object_radius(config.SMALL_OBJECT_RADIUS)
            self.server.pure_pursuit_navigator_slow.set_true_max_speed(config.SLOW_EDGE_BALL_MAX_SPEED)
        else:
            self.server.path_planner.set_object_radius(config.LARGE_OBJECT_RADIUS)
            self.server.pure_pursuit_navigator_slow.set_true_max_speed(config.SLOW_MAX_SPEED)

        # open claw 
        logger.debug("Opening claw to collect the ball.")
        instruction = {"cmd": "claw", "action": "open"}
        self.server.send_instruction(instruction)

        grid = self.server.path_planner.generate_grid(self.server.course, excluded_objects=[self.target_object])
        current_path = self.server.path_planner.find_path(self.robot_center, self.target_object.center, grid)
        if current_path is not None and len(current_path) > 0:
            self.server.pure_pursuit_navigator_slow.set_path(current_path)
            logger.info(f"Path found to collect the ball: {len(current_path)} points.")
        else:
            logger.error("No path found to collect the ball, switching to idle state.")
            from .StateIdle import StateIdle
            self.server.set_state(StateIdle(self.server))

    def update(self, frame):
        robot = super().get_last_valid_robot() # will return a valid robot, or go to idle state if not found
        self.robot_center = robot.center
        self.robot_direction = robot.direction

        instruction = self.server.pure_pursuit_navigator_slow.compute_drive_command(self.robot_center, self.robot_direction)
        self.server.path_planner_visualizer.draw_path(frame, self.server.pure_pursuit_navigator_slow.path)
        self.server.send_instruction(instruction)
        logger.debug(f"Sending drive command: {instruction}")
        #stop_dist = 20 if self.is_edge_ball else 20
        stop_dist = config.BALL_STOP_DISTANCE
        if self.is_somewhat_near_cross:
            logger.debug(f"Robot is somewhat near a cross, adjusting stop distance to {config.BALL_STOP_DISTANCE_CROSS}")
            stop_dist = config.BALL_STOP_DISTANCE_CROSS
        if self.is_edge_ball:
            logger.debug(f"Robot is near an edge ball, adjusting stop distance to {config.BALL_STOP_DISTANCE_EDGE_BALL}")
            stop_dist = config.BALL_STOP_DISTANCE_EDGE_BALL

        distance = self._distance(self.robot_center, self.server.pure_pursuit_navigator_slow.path[-1])
        logger.debug(f"Distance to target object: {distance}, stop distance: {stop_dist}, is_edge_ball: {self.is_edge_ball}")
        if distance < stop_dist:
            logger.info("Reached the end of the path to collect the ball.")
            if not self.is_edge_ball:
                instruction = {"cmd": "claw", "action": "close"}
                self.server.send_instruction(instruction)
                self.server.pure_pursuit_navigator_slow.set_path(None)
                instruction = {"cmd": "drive", "left_speed": 0, "right_speed": 0}
                self.server.send_instruction(instruction)
            if self.is_edge_ball:
                '''
                    if cmd == "close_claw_and_back":
        move_speed = instr.get("move_speed", 10)
        claw_speed = instr.get("claw_speed", 20)
        robot.close_claw_and_back(move_speed, claw_speed)
        return True
                '''
                instruction = {"cmd": "close_claw_and_back", "move_speed": 10, "claw_speed": 10}
                self.server.send_instruction(instruction)
                self.server.pure_pursuit_navigator_slow.set_path(None)
            
            logger.debug("Switching to DeliverBall state after collecting the ball.")
            GOAL = None
            if config.USE_MANUAL_GOAL_CENTER:
                GOAL = CourseObject(label='small_goal',
                               mask=None,
                               bbox=(0.0, 0.0, 0.0, 0.0),
                               confidence=float(1.0))
                GOAL.center = config.MANUAL_GOAL_CENTER  # Assuming a fixed goal position for simplicity
            else:
                GOAL = self.server.last_valid_large_goal
                if GOAL is None:
                    return frame  # If no goal is found, try again

            from .StateDeliverBall import StateDeliverBall
            self.server.set_state(StateDeliverBall(self.server, target_object=GOAL))
        return frame

    def on_exit(self):
        self.server.path_planner.set_object_radius(config.LARGE_OBJECT_RADIUS)  # Reset to

    def attempt_to_unstuck(self, frame):
        logger.info("Trying to unstuck: backing up and retrying.")

        if self.is_edge_ball and self._distance(self.robot_center, self.target_object.center) < 25:
            logger.info("Edge ball detected, and robot is within 25 pixels of the target object. Attempting to close claw and back off.")
            # Example: send a reverse command, or transition to a recovery state
            instruction = {"cmd": "close_claw_and_back", "move_speed": 20, "claw_speed": 10}
            self.server.send_instruction(instruction)
            time.sleep(2)
            logger.debug("Attempted to close claw and back off for edge ball. Continuing to deliver the ball.")
            GOAL = CourseObject(label='small_goal',
                               mask=None,
                               bbox=(0.0, 0.0, 0.0, 0.0),
                               confidence=float(1.0))
            GOAL.center = config.MANUAL_GOAL_CENTER  # Assuming a fixed goal position for simplicity
            from .StateDeliverBall import StateDeliverBall
            self.server.set_state(StateDeliverBall(self.server, target_object=GOAL))
            return frame
        else:
            logger.info("Not an edge ball or robot is too far from the target object. Backing up for 2 seconds and retrying.")
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
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5