import math
import numpy as np
import logging

logger = logging.getLogger(__name__)

class PurePursuitNavigator:
    def __init__(self, path, lookahead_distance=25, max_speed=120, true_max_speed=90, kp=0.3, max_turn_slowdown=0.7):
        self.path = path  # list of (x, y)
        self.lookahead_distance = lookahead_distance
        self.max_speed = max_speed
        self.true_max_speed = true_max_speed  # NEW: physical max wheel speed
        self.current_index = 0
        self.kp = kp
        self.max_turn_slowdown = max_turn_slowdown  # NEW: max slowdown factor for sharp turns

    def _distance(self, a, b):
        res = math.hypot(b[0] - a[0], b[1] - a[1])
        logger.debug(f"Computed distance from {a} to {b}: {res}")
        return res

    def _angle_to(self, src, dst):
        res = math.degrees(math.atan2(dst[1] - src[1], dst[0] - src[0]))
        logger.debug(f"Computed angle from {src} to {dst}: {res} degrees")
        return res  # returns angle in degrees, 0 is right, 90 is up, 180 is left, 270 is down

    def _normalize_angle(self, angle):
        res = ((angle + 180) % 360) - 180
        logger.debug(f"Normalized angle: {angle} to {res}")
        return res
    
    def set_true_max_speed(self, new_max_speed):
        """
        Set a new maximum speed for the robot.
        """
        self.true_max_speed = new_max_speed
        logger.debug(f"True max speed set to: {self.true_max_speed}")
    
    def set_path(self, new_path):
        """
        Set a new path and reset the current index.
        """
        self.path = new_path
        self.current_index = 0

    def find_lookahead_point(self, robot_pos):
        """
        Find the first point on the path that is at least lookahead_distance from the robot.
        """
        for i in range(self.current_index, len(self.path)):
            point = self.path[i]
            if self._distance(robot_pos, point) >= self.lookahead_distance:
                self.current_index = i
                logger.debug(f"Lookahead point found at index {i}: {point}")
                return point
        res = self.path[-1]  # fallback to last point if none far enough
        logger.debug(f"No lookahead point found, returning last point: {res}")
        return res

    def compute_drive_command(self, robot_pos, robot_heading):
        lookahead = self.find_lookahead_point(robot_pos)
        lookahead = (float(lookahead[0]), float(lookahead[1]))

        # Angle from robot to lookahead
        angle_to_target = self._angle_to(robot_pos, lookahead)
        logger.debug(f"Angle to target: {angle_to_target}")
        logger.debug(f"Robot Position: {robot_pos}, Lookahead: {lookahead}, Robot Heading: {robot_heading}")
        heading_error = self._normalize_angle(angle_to_target - robot_heading)

        # Steering control
        steering = -self.kp * heading_error  # NEGATE if needed for correct direction

        # Speed adjustment (slow down on sharp turns)
        slowdown_factor = 1 - min(abs(steering) / 180, self.max_turn_slowdown)
        forward_speed = self.max_speed * slowdown_factor

        left_speed = int(forward_speed - steering)
        right_speed = int(forward_speed + steering)

        # Clamp motor speeds to true physical max
        left_speed = max(-self.true_max_speed, min(self.true_max_speed, left_speed))
        right_speed = max(-self.true_max_speed, min(self.true_max_speed, right_speed))
        '''
        logger.warning(f"Robot Position: {robot_pos}, "
                     f"Heading Error: {heading_error}, "
                     f"forward_speed: {forward_speed}, "
                     f"Steering: {steering}, "
                     f"slowdown_factor: {slowdown_factor}, "
                     f"Left Speed: {left_speed}, Right Speed: {right_speed}"
                     f"Lookahead: {lookahead}, "
                     f"Robot Heading: {robot_heading}, "
                     f"Angle to Target: {angle_to_target}, ")
        '''
        logger.warning(f"slowdown_factor = 1 - min({abs(steering) / 180}, {self.max_turn_slowdown}) = {slowdown_factor}, "
                    f"Heading Error: {heading_error}, "
                     f"forward_speed: min({forward_speed}, "
                     f"Steering: {steering}, "
                     f"Left Speed: {left_speed}, Right Speed: {right_speed}"
                     f"Lookahead: {lookahead}, "
                     f"Robot Heading: {robot_heading}, "
                     f"Angle to Target: {angle_to_target}, ")

        return {"cmd": "drive", "left_speed": left_speed, "right_speed": right_speed}
    
    def compute_turn_command(self, robot_heading, target_heading, newKp = None, new_max_speed = None):
        """
        Turn in place to face target_heading (degrees),
        without any forward/backward motion.
        """
        # 1) Compute the shortest angular difference to target
        logger.debug(f"Robot Heading: {robot_heading}, Target Heading: {target_heading}")
        heading_error = self._normalize_angle(target_heading - robot_heading)

        # 2) P-control law for turning
        turn_speed = -self.kp * heading_error
        if newKp is not None:
            turn_speed = -newKp * heading_error

        logger.debug(f"Computed turn speed before clamping: {turn_speed}")

        # 3) Clamp to physical wheel limits
        turn_speed = max(-self.true_max_speed,
                        min(self.true_max_speed, turn_speed))
        if new_max_speed is not None:
            turn_speed = max(-new_max_speed,
                            min(new_max_speed, turn_speed))
            
        logger.debug(f"Turn speed after clamping: {turn_speed}")

        # 4) Opposite wheel speeds for in-place rotation
        left_speed  = int(-turn_speed)
        right_speed = int( turn_speed)

        logger.debug(f"Turn Command: Left Speed: {left_speed}, Right Speed: {right_speed}, "
                     f"Heading Error: {heading_error}, "
                        f"Turn Speed: {turn_speed}, "
                        f"New Kp: {newKp}, New Max Speed: {new_max_speed}")

        return {"cmd": "drive",
                "left_speed": left_speed,
                "right_speed": right_speed}
