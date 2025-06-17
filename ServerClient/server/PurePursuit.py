import math
import numpy as np

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
        return math.hypot(b[0] - a[0], b[1] - a[1])

    def _angle_to(self, src, dst):
        return math.degrees(math.atan2(dst[1] - src[1], dst[0] - src[0]))

    def _normalize_angle(self, angle):
        return ((angle + 180) % 360) - 180
    
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
                return point
        return self.path[-1]  # fallback to last point if none far enough

    def compute_drive_command(self, robot_pos, robot_heading):
        lookahead = self.find_lookahead_point(robot_pos)
        lookahead = (float(lookahead[0]), float(lookahead[1]))

        # Angle from robot to lookahead
        angle_to_target = self._angle_to(robot_pos, lookahead)
        heading_error = self._normalize_angle(angle_to_target - robot_heading)

        # Steering control
        steering = -self.kp * heading_error  # NEGATE if needed for correct direction

        # Speed adjustment (slow down on sharp turns)
        forward_speed = self.max_speed * (1 - min(abs(steering) / 100, self.max_turn_slowdown))

        left_speed = int(forward_speed - steering)
        right_speed = int(forward_speed + steering)

        # Clamp motor speeds to true physical max
        left_speed = max(-self.true_max_speed, min(self.true_max_speed, left_speed))
        right_speed = max(-self.true_max_speed, min(self.true_max_speed, right_speed))

        print(f"[DEBUG] Robot Position: {robot_pos}, Lookahead: {lookahead}, "
              f"Robot Heading: {robot_heading}, "
              f"Angle to Target: {angle_to_target}, Heading Error: {heading_error}, " 
              f"Left Speed: {left_speed}, Right Speed: {right_speed}")

        return {"cmd": "drive", "left_speed": left_speed, "right_speed": right_speed}
    
    def compute_turn_command(self, robot_heading, target_heading):
        """
        Turn in place to face target_heading (degrees),
        without any forward/backward motion.
        """
        # 1) Compute the shortest angular difference to target
        heading_error = self._normalize_angle(target_heading - robot_heading)

        # 2) P-control law for turning
        turn_speed = self.kp * heading_error

        # 3) Clamp to physical wheel limits
        turn_speed = max(-self.true_max_speed,
                         min(self.true_max_speed, turn_speed))

        # 4) Opposite wheel speeds for in-place rotation
        left_speed  = int(-turn_speed)
        right_speed = int( turn_speed)

        print(f"[DEBUG] Heading Error: {heading_error:.1f}°, "
              f"Turn Speed: {turn_speed:.1f}, "
              f"→ Left: {left_speed}, Right: {right_speed}")

        return {"cmd": "drive",
                "left_speed": left_speed,
                "right_speed": right_speed}
