from .StateBase import StateBase

import logging
logger = logging.getLogger(__name__)

class StateGoToNearestBall(StateBase):
    def __init__(self, server):
        super().__init__(server)
        logger.debug("Initialized StateGoToNearestBall.")

    def on_enter(self):
        logger.debug("Entering StateGoToNearestBall.")
        # Determine the nearest ball that is not at a corner or cross
        robot = super().get_last_valid_robot() # will return a valid robot, or go to idle state if not found
            
        self.robot_center = robot.center
        self.robot_direction = robot.direction

        all_balls = self.server.course.get_white_balls() + self.server.course.get_orange_balls()
        logger.debug(f"Found {len(all_balls)} balls in total.")
        all_balls = self._sort_balls_by_distance(self.robot_center, all_balls)
        logger.debug(f"Balls sorted by distance: {[ball.center for ball in all_balls]}")

        for ball in all_balls:
            near_cross = self.server.course.is_ball_near_cross(ball)
            near_corner = self.server.course.is_ball_near_corner(ball)
            logger.debug(f"Checking ball at {ball.center}: near_corner={near_corner}, near_cross={near_cross}")
            if near_corner:
                logger.info(f"Skipping ball {ball} as it is near a corner.")
                continue  # Skip balls that are near corners or crosses
            if near_cross:
                logger.info(f"Skipping ball {ball} as it is near a cross.")
                continue

            # check if a path can be generated to this ball
            optimal_spot = self.server.course.get_optimal_ball_parking_spot(ball)
            logger.debug(f"Optimal parking spot for ball {ball}: {optimal_spot}")
            if optimal_spot is None:
                logger.warning(f"No optimal parking spot found for ball: {ball}. Trying next ball...")
                continue
            
            # try to generate a path to the ball
            grid = self.server.path_planner.generate_grid(self.server.course)
            logger.debug(f"Generated grid for ball {ball}.")
            path = self.server.path_planner.find_path(self.robot_center, optimal_spot, grid)
            logger.debug(f"Path for ball {ball}, length: {len(path) if path else 'None'}")
            if path is None or len(path) == 0:
                logger.warning(f"No path found to ball: {ball}. Trying to find another ball...")
                continue
        
            # If we reach here, we have a valid path to the ball
            self.server.pure_pursuit_navigator.set_path(path)
            logger.info(f"Best ball found: {ball}. Path generated.")
            self.target_ball = ball
            logger.debug(f"Set target_ball to {ball}")
            break  # Stop after finding the first valid ball
        
        if self.server.pure_pursuit_navigator.path is None or len(self.server.pure_pursuit_navigator.path) == 0:
            logger.warning("No valid path found to any ball. Please try again.")
            from .StateIdle import StateIdle
            self.server.set_state(StateIdle(self.server))

    def update(self, frame):
        logger.debug("Updating StateGoToNearestBall.")
        if len(self.server.pure_pursuit_navigator.path) == 0:
            logger.warning("No path to follow")
            return
        
        if self.server.course.get_robot() is not None:
            robot = self.server.course.get_robot()
            self.robot_center = robot.center
            self.robot_direction = robot.direction
            logger.debug(f"Updated robot_center: {self.robot_center}, robot_direction: {self.robot_direction}")
        else:
            logger.warning("No robot found in the course, using previous position.")
        
        frame = self.server.path_planner_visualizer.draw_path(frame, self.server.pure_pursuit_navigator.path)
        logger.debug("Path drawn on frame.")
        instruction = self.server.pure_pursuit_navigator.compute_drive_command(self.robot_center, self.robot_direction)
        logger.debug(f"Computed drive instruction: {instruction}")
        self.server.send_instruction(instruction)
        logger.debug("Instruction sent to robot.")
        
        if self._distance(self.robot_center, self.server.pure_pursuit_navigator.path[-1]) < 10:
            logger.info("Reached the end of the path.")            
            # Go to idle state after reaching the ball
            from .StateIdle import StateIdle
            from .StateRotateToObject import StateRotateToObject
            logger.debug(f"Switching to StateRotateToObject with target_object={self.target_ball}")
            self.server.set_state(StateRotateToObject(self.server, target_object=self.target_ball))

    def on_exit(self):
        logger.debug("Exiting StateGoToNearestBall. Clearing path.")
        self.server.pure_pursuit_navigator.set_path(None)

    def on_click(self, event, x, y):
        logger.debug(f"on_click at ({x}, {y})")
        return super().on_click(x, y)
    
    def on_key_press(self, key):
        logger.debug(f"on_key_press called with key={key}")
        if key == ord('x'):
            # If 'x' is pressed, go back to idle state
            from .StateIdle import StateIdle
            logger.info("'x' pressed, switching to StateIdle.")
            self.server.set_state(StateIdle(self.server))
    
    def _distance(self, a, b):
        """
        Calculate the Euclidean distance between two points.
        """
        d = ((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) ** 0.5
        logger.debug(f"Calculated distance between {a} and {b}: {d}")
        return d

    def _sort_balls_by_distance(self, robot_center, balls):
        """
        Sort balls by distance from the robot so balls[0] is the nearest ball.
        """
        sorted_balls = sorted(balls, key=lambda ball: self._distance(robot_center, ball.center))
        logger.debug(f"Sorted balls by distance: {[ball.center for ball in sorted_balls]}")
        return sorted_balls