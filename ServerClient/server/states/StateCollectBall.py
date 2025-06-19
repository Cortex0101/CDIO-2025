from .StateBase import StateBase
import config

import time

class StateCollectBall(StateBase):
    def __init__(self, server, target_object=None):
        super().__init__(server)
        self.target_object = target_object

    def on_enter(self):
        self.robot_direction = self.server.course.get_robot().direction
        self.robot_center = self.server.course.get_robot().center
        self.is_edge_ball = self.server.course.is_ball_near_wall(self.target_object)

        ### quick fix, as it does not recognize the target object, passed from earlier ###
        new_target = self.server.course.get_nearest_ball(self.robot_center, color='white')
        if new_target is not None:
            self.target_object = new_target
        ###################

        if self.is_edge_ball:
            self.server.path_planner.set_object_radius(config.SMALL_OBJECT_RADIUS)
        else:
            self.server.path_planner.set_object_radius(config.LARGE_OBJECT_RADIUS)

        # open claw 
        instruction = {"cmd": "claw", "action": "open"}
        self.server.send_instruction(instruction)

        grid = self.server.path_planner.generate_grid(self.server.course, excluded_objects=[self.target_object])
        current_path = self.server.path_planner.find_path(self.robot_center, self.target_object.center, grid)
        if current_path is not None and len(current_path) > 0:
            self.server.pure_pursuit_navigator_slow.set_path(current_path)
            print(f"[SERVER] Path found to collect the ball: {len(current_path)} points.")
        else:
            print("[SERVER] No path found to collect the ball.")
            from .StateIdle import StateIdle
            self.server.set_state(StateIdle(self.server))

    def update(self, frame):
        if self.server.course.get_robot() is not None:
            robot = self.server.course.get_robot()
            self.robot_center = robot.center if robot.center is not None else self.robot_center
            self.robot_direction = robot.direction if robot.direction is not None else self.robot_direction
        else:
            print("[SERVER] No robot found in the course, using previous position.")

        instruction = self.server.pure_pursuit_navigator_slow.compute_drive_command(self.robot_center, self.robot_direction)
        self.server.path_planner_visualizer.draw_path(frame, self.server.pure_pursuit_navigator_slow.path)
        self.server.send_instruction(instruction)
        stop_dist = 10 if self.is_edge_ball else 20

        if self._distance(self.robot_center, self.server.pure_pursuit_navigator_slow.path[-1]) < stop_dist:
            print("[SERVER] Reached the end of the path.")
            instruction = {"cmd": "claw", "action": "close"}
            self.server.send_instruction(instruction)
            self.server.pure_pursuit_navigator_slow.set_path(None)
            instruction = {"cmd": "drive", "left_speed": 0, "right_speed": 0}
            self.server.send_instruction(instruction)
            time.sleep(0.5)  # Wait for claw to close
            if self.is_edge_ball:
                instruction = {"cmd": "drive_seconds", "seconds": 2, "speed": -10}
                self.server.send_instruction(instruction)
                time.sleep(2)  # Back off a bit for edge balls
            
            from .StateIdle import StateIdle
            self.server.set_state(StateIdle(self.server))

    def on_exit(self):
        self.server.path_planner.set_object_radius(config.LARGE_OBJECT_RADIUS)  # Reset to


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