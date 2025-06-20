import StateBase
import logging

logger = logging.getLogger(__name__)

class StateDeliverBall(StateBase):
    def __init__(self, server, target_object=None): # might not need target_object as its just the closest ball to the robot that should be ignored
        self.server = server  # Reference to the main Server object

    def update(self, frame):
        """
        Update the state with the current frame.
        This method is called periodically to update the state.
        """
        if self.server.course.get_robot() is not None:
            self.robot_center = self.server.course.get_robot().center
            self.robot_direction = self.server.course.get_robot().direction
        else:
            logger.error("[SERVER] No robot found in the course, using previous position.")
        
        frame = self.server.path_planner_visualizer.draw_path(frame, self.server.pure_pursuit_navigator.path)
        instruction = self.server.pure_pursuit_navigator.compute_drive_command(self.robot_center, self.robot_direction)
        self.server.send_instruction(instruction)
        
        if self._distance(self.robot_center, self.server.pure_pursuit_navigator.path[-1]) < 10:
            logger.info("[SERVER] Reached the end of the path.")
            self.server.pure_pursuit_navigator.set_path(None)
            instruction = {"cmd": "drive", "left_speed": 0, "right_speed": 0}
            self.server.send_instruction(instruction)

    def on_enter(self):
        self.GOAL_LOCATION = (350, 450)  # HARD-CODED GOAL LOCATION, replace with self.course.get_goals find the closest
        self.robot = self.server.course.get_robot()
        if self.robot is None:
            logger.error("[SERVER] No robot found in the course, returning to idle state.")
            from .StateIdle import StateIdle
            self.server.set_state(StateIdle(self.server))
        self.robot_direction = self.robot.direction()
        self.robot_center = self.robot.center
        
        self.target_location = self.server.course.get_optimal_goal_parking_spot(self.GOAL_LOCATION)
        excluded_ball = self.server.course.get_nearest_ball(self.course.get_robot().center, 'either')
        current_path = self.server.path_planner.find_path(self.robot_center, self.target_location, self.path_planner.generate_grid(self.course, excluded_objects=[excluded_ball] if excluded_ball is not None else []))
        if current_path is not None and len(current_path) > 0:
            self.server.pure_pursuit_navigator.set_path(current_path)
            logger.info(f"[SERVER] Path found to deliver the ball: {len(current_path)} points.")
        else:
            logger.error("[SERVER] No path found to deliver the ball.")
            from .StateIdle import StateIdle
            self.server.set_state(StateIdle(self.server))

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