from .StateBase import StateBase

import logging
logger = logging.getLogger(__name__)

class StateIdle(StateBase):
    def update(self, frame):
        # look for robot all the time
        super().get_last_valid_robot()
        return frame

    def on_enter(self):
        logger.debug("Entered idle state")
        instruction = {"cmd": "drive", "left_speed": 0, "right_speed": 0}
        self.server.send_instruction(instruction)
        instruction = {"cmd": "claw", "action": "close"}
        self.server.send_instruction(instruction)

    def on_exit(self):
        pass

    def on_click(self, event, x, y):
        """
        Handle click events. Default implementation does nothing.
        Override in subclasses if needed.
        """
        logger.info(f"Clicked at ({x}, {y}) in idle state")

    def on_key_press(self, key):
        if key == ord('g'):
            # Start StateGotoNearestBall
            from .StateGoToNearestBall import StateGoToNearestBall
            logger.info("Key 'g' pressed, switching to StateGoToNearestBall.")
            self.server.set_state(StateGoToNearestBall(self.server))
        
        elif key == ord('b'):
            nearest_ball = self.server.course.get_nearest_ball(self.server.course.get_robot().center, color='white')
            logger.info(f"B pressed, nearest ball: {nearest_ball}")
            if nearest_ball is not None:
                from .StateCollectBall import StateCollectBall
                logger.info("Switching to StateCollectBall to collect the nearest ball.")
                self.server.set_state(StateCollectBall(self.server, target_object=nearest_ball))

        elif key == ord('d'):
            # Start StateDeliverBall
            from .StateDeliverBall import StateDeliverBall
            logger.info("Key 'd' pressed, switching to StateDeliverBall.")
            self.server.set_state(StateDeliverBall(self.server))
