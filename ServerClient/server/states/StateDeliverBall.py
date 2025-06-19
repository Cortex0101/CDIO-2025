import StateBase
import logging

logger = logging.getLogger(__name__)

class StateDeliverBall(StateBase):
    def __init__(self, server):
        self.server = server  # Reference to the main Server object

    def update(self, frame):
        pass

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
        if key == ord('x'):
            # If 'g' is pressed, go back to idle state
            from .StateIdle import StateIdle
            self.server.set_state(StateIdle(self.server))

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