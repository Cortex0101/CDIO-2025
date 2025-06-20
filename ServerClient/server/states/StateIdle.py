from .StateBase import StateBase

class StateIdle(StateBase):
    def update(self, frame):
        pass

    def on_enter(self):
        instruction = {"cmd": "drive", "left_speed": 0, "right_speed": 0}
        self.server.send_instruction(instruction)

    def on_exit(self):
        pass

    def on_click(self, event, x, y):
        """
        Handle click events. Default implementation does nothing.
        Override in subclasses if needed.
        """
        pass

    def on_key_press(self, key):
        if key == ord('g'):
            # Start StateGotoNearestBall
            from .StateGoToNearestBall import StateGoToNearestBall
            self.server.set_state(StateGoToNearestBall(self.server))
        
        elif key == ord('b'):
            nearest_ball = self.server.course.get_nearest_ball(self.server.course.get_robot().center, color='white')
            if nearest_ball is not None:
                from .StateCollectBall import StateCollectBall
                self.server.set_state(StateCollectBall(self.server, target_object=nearest_ball))
