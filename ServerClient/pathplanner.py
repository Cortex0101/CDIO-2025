import math

def compute_motor_speeds(robot_x, robot_y, robot_angle_deg, target_x, target_y, base_speed=100, max_diff=100):
    """
    Calculate left and right motor speeds for a differential drive robot
    to move toward a target point smoothly.

    Parameters:
        robot_x, robot_y: Current robot position
        robot_angle_deg: Robot heading in degrees (0 = right, 90 = up)
        target_x, target_y: Target position
        base_speed: Max forward speed when facing target
        max_diff: Maximum speed reduction on the inner wheel when turning

    Returns:
        Dict with 'left_speed' and 'right_speed'
    """
    # Calculate angle to target
    dx = target_x - robot_x
    dy = target_y - robot_y
    target_angle_rad = math.atan2(dy, dx)
    target_angle_deg = math.degrees(target_angle_rad)

    # Normalize both angles to [0, 360)
    target_angle_deg = target_angle_deg % 360
    robot_angle_deg = robot_angle_deg % 360

    # Compute smallest signed angle difference
    angle_diff = (target_angle_deg - robot_angle_deg + 540) % 360 - 180  # Range [-180, 180]

    # Calculate turn factor (from -1 to 1)
    turn_factor = angle_diff / 90  # Full steering if 90° off
    turn_factor = max(-1, min(1, turn_factor))  # Clamp to [-1, 1]

    # Calculate motor speeds based on turn factor
    left_speed = base_speed
    right_speed = base_speed

    if turn_factor > 0:
        # Turn right (reduce right speed)
        right_speed -= abs(turn_factor) * max_diff
    else:
        # Turn left (reduce left speed)
        left_speed -= abs(turn_factor) * max_diff

    return {
        'left_speed': int(left_speed),
        'right_speed': int(right_speed)
    }

if __name__ == "__main__":
    # test the function with example values

    # robot is facing right (0 degrees)
    robot_x = 0
    robot_y = 0
    robot_angle_deg = 0

    # target is directly above the robot (90 degrees)
    target_x = 0
    target_y = 10
    speeds = compute_motor_speeds(robot_x, robot_y, robot_angle_deg, target_x, target_y)
    print('Testing with target above robot:')
    print(f"Left Speed: {speeds['left_speed']}, Right Speed: {speeds['right_speed']}")

    # test with target to the right (0 degrees)
    target_x = 10
    target_y = 0
    speeds = compute_motor_speeds(robot_x, robot_y, robot_angle_deg, target_x, target_y)
    print('Testing with target to the right:')
    print(f"Left Speed: {speeds['left_speed']}, Right Speed: {speeds['right_speed']}")

    # test with target to the left (180 degrees)
    target_x = -10
    target_y = 0
    speeds = compute_motor_speeds(robot_x, robot_y, robot_angle_deg, target_x, target_y)
    print('Testing with target to the left:')
    print(f"Left Speed: {speeds['left_speed']}, Right Speed: {speeds['right_speed']}")