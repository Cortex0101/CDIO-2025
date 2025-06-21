import logging
from collections import deque
import time
logger = logging.getLogger(__name__)

class StateBase:
    STUCK_HISTORY_LENGTH = 30  # Number of frames to keep
    STUCK_TIME_SEC = 10        # Time window to check for stalling
    STUCK_DIST_THRESHOLD = 5   # Pixels
    STUCK_ANGLE_THRESHOLD = 5  # Degrees

    def __init__(self, server):
        self.server = server
        self._stuck_history = deque(maxlen=self.STUCK_HISTORY_LENGTH)
        self._last_unstuck_time = time.time()
        
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
        # Call this instead of update in main loop
        robot = self.server.course.get_robot()
        if robot:
            now = time.time()
            self._stuck_history.append((robot.center, robot.direction, now))
        if self.is_stuck():
            logger.warning("Robot appears to be stuck. Attempting to unstuck.")
            return self.attempt_to_unstuck(frame)
        else:
            return self.update(frame)


    # dont allow override
    def get_last_valid_robot(self):
        """
        Returns the last valid robot position.
        This can be overridden in subclasses if needed.
        """
        robot = self.server.course.get_robot()
        robot_direction = robot.direction if robot else None
        robot_center = robot.center if robot else None

        if robot is not None and robot_direction is not None and robot_center is not None:
            self.server.last_valid_robot = robot
            return robot
        else:
            logger.warning("No valid robot found in the course. Using last valid robot.")
            if self.server.last_valid_robot is None:
                logger.error("No previous valid robot found. Cannot proceed, going to idle state.")
                from .StateIdle import StateIdle
                self.server.set_state(StateIdle(self.server))
                return None
            else:
                logger.warning(f"Using last valid robot: {self.server.last_valid_robot}")
                return self.server.last_valid_robot

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
        
        