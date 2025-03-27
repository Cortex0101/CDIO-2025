
from ev3dev2.motor import LargeMotor, MoveSteering, OUTPUT_A, OUTPUT_B
from ev3dev2.sound import Sound
from ev3dev2.sensor.lego import UltrasonicSensor
from ev3dev2.sensor import INPUT_1
import math

ultrasonic = UltrasonicSensor(INPUT_1)

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
    sorted_points = sort_proximity(robot.get_position(), target_points)
    current_x, current_y = robot.get_position()
    current_heading = robot.get_angle()

    for target_x, target_y in sorted_points:
        dx = target_x - current_x
        dy = target_y - current_y
        distance = calculate_distance((current_x, current_y), (target_x, target_y))
        
        # Correct calculation of target heading
        target_heading = math.degrees(math.atan2(dy, dx))
        
        # Calculate the shortest turn direction
        turn_angle = (target_heading - current_heading + 180) % 360 - 180

        print("From Pathfinding, move_robot()")
        print("Moving to: " + str(target_x) + ", " + str(target_y) +
              ", Turn angle: " + str(turn_angle) + ", Distance: " + str(distance))

        # Perform turn (correcting left/right logic)
        if turn_angle > 0:
            robot.turn_left(turn_angle)  # Left turn if positive
        elif turn_angle < 0:
            robot.turn_right(-turn_angle)  # Right turn if negative

        while ultrasonic.distance_centimeters < 5:
            # move backwards or stop -> logic
            robot.move_backward(5)
            

        # Move forward
        robot.move_forward(distance)

        # Update current position and heading
        current_x, current_y = target_x, target_y
        current_heading = (current_heading + turn_angle) % 360

    print("Navigation completed")
