import logging
from collections import deque
import time
logger = logging.getLogger(__name__)

class StateBase:
    STUCK_HISTORY_LENGTH = 100  # Number of frames to keep
    STUCK_TIME_SEC = 7        # Time window to check for stalling
    STUCK_DIST_THRESHOLD = 5   # Pixels
    STUCK_ANGLE_THRESHOLD = 5  # Degrees
    CAPTURE_INTERVAL = 0.1  # Interval to capture frames for unstuck logic

    def __init__(self, server):
        self.server = server
        self._stuck_history = deque(maxlen=int(self.STUCK_HISTORY_LENGTH * 1.1)) # slightly larger to make sure we have enough history to cover STUCK_TIME_SEC 
        self._last_unstuck_time = time.time()
        self._last_stuck_history_time = time.time()

        self.HAS_FOUND_GOAL = False # Only find goal ONCE

        
    def on_enter(self):
        pass

    def update(self, *args, **kwargs):
        raise NotImplementedError("Each state must implement its own update method.")

    def on_exit(self):
        pass

    def attempt_to_unstuck(self, frame):
        logger.warning("Default attempt_to_unstuck called. Override in subclass for custom behavior.")
        # Optionally, transition to a recovery state or try a default action
        return frame

    def on_click(self, event, x, y):
        """
        Handle click events. Default implementation does nothing.
        Override in subclasses if needed.
        """
        pass

    def on_key_press(self, key):
        """
        Handle key press events. Default implementation does nothing.
        Override in subclasses if needed.
        """
        pass

    def is_stuck(self):
        # Not enough history yet
        if len(self._stuck_history) < self.STUCK_HISTORY_LENGTH:
            return False
        # Check if position/direction changed significantly in last STUCK_TIME_SEC
        first = self._stuck_history[0]
        last = self._stuck_history[-1]
        dt = last[2] - first[2]
        if dt < self.STUCK_TIME_SEC:
            return False
        dist = ((last[0][0] - first[0][0]) ** 2 + (last[0][1] - first[0][1]) ** 2) ** 0.5
        angle_diff = abs(last[1] - first[1])
        if dist < self.STUCK_DIST_THRESHOLD and angle_diff < self.STUCK_ANGLE_THRESHOLD:
            return True
        return False

    def step(self, frame):
        robot = self.get_last_valid_robot()
        if not robot:
            return frame  # If no valid robot, return the frame without further processing
        large_goal = self.get_last_valid_large_goal()
        if not large_goal:
            return frame
        
        self.server.last_valid_robot = robot  # Update last valid robot
        self.server.last_valid_large_goal = self.get_last_valid_large_goal()  # Update last valid large goal
        now = time.time()
        # Only store if enough time has passed since last entry
        if now - self._last_stuck_history_time > self.CAPTURE_INTERVAL:
            self._stuck_history.append((robot.center, robot.direction, now))
            self._last_stuck_history_time = now
        if self.is_stuck():
            return self.attempt_to_unstuck(frame)
        else:
            return self.update(frame)

    def get_last_valid_large_goal(self):
        """
        Returns the last valid large goal position.
        If the goal is detected but its center is None, use the new goal's center
        but copy the position from the previous valid goal.
        """
        if self.HAS_FOUND_GOAL:
            return self.server.last_valid_large_goal

        goal = self.server.course.get_large_goal()
        last_valid = self.server.last_valid_large_goal

        if goal is not None:
            if goal.center is not None:
                self.server.last_valid_large_goal = goal
                self.HAS_FOUND_GOAL = True
                return goal
            elif last_valid is not None:
                # Copy position from last valid goal
                logger.warning("Goal center is None, copying position from last valid goal.")
                goal.center = last_valid.center
                self.server.last_valid_large_goal = goal
                return goal
            else:
                logger.warning("Goal detected but missing center and no last valid goal to copy from.")
        else:
            logger.warning("No valid large goal found in the course. Using last valid large goal.")

        if last_valid is None:
            logger.error("No previous valid large goal found. Cannot proceed, going to idle state.")
            from .StateIdle import StateIdle
            self.server.set_state(StateIdle(self.server))
            return None
        else:
            logger.warning(f"Using last valid large goal: {last_valid}")
            return last_valid

    # dont allow override
    def get_last_valid_robot(self):
        """
        Returns the last valid robot position.
        If the robot is detected but its direction is None, use the new robot's center
        but copy the direction from the previous valid robot.
        """
        robot = self.server.course.get_robot()
        last_valid = self.server.last_valid_robot

        if robot is not None:
            if robot.direction is not None and robot.center is not None:
                self.server.last_valid_robot = robot
                return robot
            elif last_valid is not None and robot.center is not None:
                # Copy direction from last valid robot
                #logger.warning("Robot direction is None, copying direction from last valid robot.")
                robot.direction = last_valid.direction
                self.server.last_valid_robot = robot
                return robot
            else:
                pass
                #logger.warning("Robot detected but missing direction and no last valid robot to copy from.")
        else:
            pass
            #logger.warning("No valid robot found in the course. Using last valid robot.")

        if last_valid is None:
            #logger.error("No previous valid robot found. Cannot proceed, going to idle state.")
            from .StateIdle import StateIdle
            self.server.set_state(StateIdle(self.server))
            return None
        else:
            #logger.warning(f"Using last valid robot: {last_valid}")
            return last_valid

        '''
        if robot is None:
            logger.warning("No robot found in the course. Using self.server.last_valid_robot")
            if self.server.last_valid_robot is None:
                logger.error("No previous valid robot found. Cannot proceed, going to idle state.")
                from .StateIdle import StateIdle
                self.server.set_state(StateIdle(self.server))
                return None
            else:
                logger.warning(f"Robot in current frame not available, using last valid robot: {self.server.last_valid_robot}")
                return self.server.last_valid_robot
        else:
            self.server.last_valid_robot = robot  # Update last valid robot
            return self.server.last_valid_robot
        '''
        
        