from .StateBase import StateBase

class StateGoToNearestBall(StateBase):
    def __init__(self, server):
        super().__init__(server)

    def on_enter(self):
        # Determine the nearest ball that is not at a corner or cross
        self.robot_center = self.server.course.get_robot().center
        self.robot_direction = self.server.course.get_robot().direction

        all_balls = self.server.course.get_white_balls() + self.server.course.get_orange_balls()
        all_balls = self._sort_balls_by_distance(self.robot_center, all_balls)

        for ball in all_balls:
            if self.server.course.is_ball_near_corner(ball) or self.server.course.is_ball_near_cross(ball):
                continue  # Skip balls that are near corners or crosses

            # check if a path can be generated to this ball
            optimal_spot = self.server.course.get_optimal_ball_parking_spot(ball)
            if optimal_spot is None:
                print(f"[SERVER] No optimal parking spot found for ball: {ball}. Trying next ball...")
                continue
            
            # try to generate a path to the ball
            grid = self.server.path_planner.generate_grid(self.server.course)
            path = self.server.path_planner.find_path(self.robot_center, optimal_spot, grid)
            if path is None or len(path) == 0:
                print("[SERVER] No path found to ball: {ball}. Trying to find another ball...")
                continue
        
            # If we reach here, we have a valid path to the ball
            self.server.pure_pursuit_navigator.set_path(path)
            print(f"[SERVER] Best ball found: {ball}. Path generated.")
            self.target_ball = ball
        
        if self.server.pure_pursuit_navigator.path is None or len(self.server.pure_pursuit_navigator.path) == 0:
            print("[SERVER] No valid path found to any ball. Please try again.")
            from .StateIdle import StateIdle
            self.server.set_state(StateIdle(self.server))

    def update(self, frame):
        if len(self.server.pure_pursuit_navigator.path) == 0:
            print("[SERVER] No path to follow")
            return
        
        if self.server.course.get_robot() is not None:
            robot = self.server.course.get_robot()
            self.robot_center = robot.center
            self.robot_direction = robot.direction
        else:
            print("[SERVER] No robot found in the course, using previous position.")
        
        frame = self.server.path_planner_visualizer.draw_path(frame, self.server.pure_pursuit_navigator.path)
        instruction = self.server.pure_pursuit_navigator.compute_drive_command(self.robot_center, self.robot_direction)
        self.server.send_instruction(instruction)
        
        if self._distance(self.robot_center, self.server.pure_pursuit_navigator.path[-1]) < 10:
            print("[SERVER] Reached the end of the path.")            
            # Go to idle state after reaching the ball
            from .StateIdle import StateIdle
            from .StateRotateToObject import StateRotateToObject
            self.server.set_state(StateRotateToObject(self.server, target_object=self.target_ball))

    def on_exit(self):
        self.server.pure_pursuit_navigator.set_path(None)

    def on_click(self, event, x, y):
        return super().on_click(x, y)
    
    def on_key_press(self, key):
        if key == ord('x'):
            # If 'g' is pressed, go back to idle state
            from .StateIdle import StateIdle
            self.server.set_state(StateIdle(self.server))
    
    def _distance(self, a, b):
        """
        Calculate the Euclidean distance between two points.
        """
        return ((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) ** 0.5

    def _sort_balls_by_distance(self, robot_center, balls):
        """
        Sort balls by distance from the robot so balls[0] is the nearest ball.
        """
        return sorted(balls, key=lambda ball: self._distance(robot_center, ball.center))
    
'''
grid = self.path_planner.generate_grid(self.course) # change to True if you want to drw floor
                    grid_img = self.path_planner_visualizer.draw_grid_objects(grid)
                    start = robot.center
                    start = (int(start[0]), int(start[1]))
                    end = (int(x), int(y))
                    print(f"[SERVER] Generating path from {start} to {end}...")
                    current_path = self.path_planner.find_path(start, end, grid)
                    if current_path is None or len(current_path) == 0:
                        print("[SERVER] No path found, please try again.")
                        self.pure_pursuit_navigator.set_path(None)
                        continue
                    self.pure_pursuit_navigator.set_path(current_path)
                    print(f"[SERVER] Path found: {len(current_path)} points.")
                    cv2.imshow("grid_visualization", grid_img)

if (self.pure_pursuit_navigator.path is not None) and current_state == RobotState.FOLLOW_PATH:
                if len(self.pure_pursuit_navigator.path) == 0:
                    print("[SERVER] No path to follow, please generate a path first.")
                    continue
                if spot is not None:
                    self.course_visualizer.highlight_point(current_video_frame_with_objs, spot, color=(0, 255, 0), radius=10)
                current_video_frame_with_objs = self.path_planner_visualizer.draw_path(current_video_frame_with_objs, current_path)
                instruction = self.pure_pursuit_navigator.compute_drive_command(robot.center, robot_direction)
                self.send_instruction(instruction)
                if distance(robot.center, current_path[-1]) < 10:
                    print("[SERVER] Reached the end of the path.")
                    self.pure_pursuit_navigator.set_path(None)
                    instruction = {"cmd": "drive", "left_speed": 0, "right_speed": 0}
                    self.send_instruction(instruction)
'''