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