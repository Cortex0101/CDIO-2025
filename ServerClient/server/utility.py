import math

def get_next_path_point(robot_pos, path, lookahead=20):
    # Find the path point at least `lookahead` pixels ahead of robot
    for pt in path:
        if compute_distance(robot_pos, pt) > lookahead:
            return pt
    return path[-1]

def compute_distance(p1, p2):
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2) ** 0.5

def compute_angle_error(robot_pos, robot_angle, target_point):
    import math
    dx = target_point[0] - robot_pos[0]
    dy = target_point[1] - robot_pos[1]
    path_angle = math.atan2(dy, dx)
    angle_error = path_angle - robot_angle
    # Normalize to [-pi, pi]
    while angle_error > math.pi:
        angle_error -= 2*math.pi
    while angle_error < -math.pi:
        angle_error += 2*math.pi
    return angle_error