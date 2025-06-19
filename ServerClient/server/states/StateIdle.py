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
        pass