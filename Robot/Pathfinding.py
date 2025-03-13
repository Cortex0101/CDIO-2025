
from ev3dev2.motor import LargeMotor, MoveSteering, OUTPUT_A, OUTPUT_B
from ev3dev2.sound import Sound
import math
from BallDetection import white_balls, orange_balls


def calculate_distance(point1, point2):
    return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5

def sort_proximity(robot_position, points):
    if not points:
        return []
   
    remaining_points = points.copy()
   
    sorted_points = []

    current_position = robot_position

    while remaining_points:

        # Calculate distances from current position to all remaining points
        distances = [(calculate_distance(current_position, point), point) for point in remaining_points]
       
        closest_distance, closest_point = min(distances)
        sorted_points.append(closest_point)
        remaining_points.remove(closest_point)
        current_position = closest_point
   
    return sorted_points

def move_robot(robot, target_points, wheel_diameter=70, axle_track=165):
    sound = Sound()


    sorted_points = sort_proximity(robot.getPosition(), target_points)
    current_x, current_y = robot.getPosition()
    
    current_heading = robot.getAngle()
    
    print(f"Robot starting at position: ({current_x}, {current_y}), heading: {current_heading}°")
    
    # Visit each point in order
    for target_x, target_y in sorted_points:
        dx = target_x - current_x
        dy = target_y - current_y
        distance = calculate_distance((current_x, current_y), (target_x, target_y))
        target_heading = math.degrees(math.atan2(dy, dx))
        turn_angle = target_heading - current_heading
        
        if turn_angle > 180:
            turn_angle -= 360
        elif turn_angle < -180:
            turn_angle += 360
        
        print(f"Moving to ({target_x}, {target_y})")
        print(f"  - Turn angle: {turn_angle}°")
        print(f"  - Distance: {distance} mm")
        
        # Turn the robot
        if turn_angle != 0:
            # if turn_angle > 0, turn right, else turn left
            if turn_angle > 0:
                robot.turn_right(turn_angle)
            else:
                robot.turn_left(-turn_angle)

        
        # Move forward
        robot.move_forward(distance)
        sound.beep()
    print("Navigation completed")