import logging
logger = logging.getLogger(__name__)

class StateBase:
    def __init__(self, server):
        self.server = server  # Reference to the main Server object

    def update(self, *args, **kwargs):
        raise NotImplementedError("Each state must implement its own update method.")

    def on_enter(self):
        pass

    def on_exit(self):
        pass

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
        
        