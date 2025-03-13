

from pybricks.hubs import EV3Brick
from pybricks.ev3devices import Motor, UltrasonicSensor, ColorSensor
from pybricks.parameters import Port, Direction
from pybricks.tools import wait
from pybricks.robotics import DriveBase
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

def move_robot(robot_position, target_points, wheel_diameter=70, axle_track=165):

    ev3 = EV3Brick()
    
    left_motor = Motor(Port.A)
    right_motor = Motor(Port.B)

    # drivebase for correct speed calcs, axle track = distance between middle of wheels in mm
    robot = DriveBase(left_motor, right_motor, wheel_diameter, axle_track)
    
    robot.settings(drive_speed=100, turn_rate=60)

    sorted_points = sort_proximity(robot_position, target_points)

    current_x, current_y = robot_position
    
    # degrees, 0 = positive x-axis, 90 = positive y-axis
    current_heading = 0
    
    print(f"Robot starting at position: ({current_x}, {current_y}), heading: {current_heading}°")
    
    # Visit each point in order
    for target_x, target_y in sorted_points:
        # Calculate displacement vector
        dx = target_x - current_x
        dy = target_y - current_y
        
        # Calculate distance to target
        distance = calculate_distance((current_x, current_y), (target_x, target_y))

        # Calculate target heading (in degrees)
        # atan2 returns angle in radians, convert to degrees
        target_heading = math.degrees(math.atan2(dy, dx))
        
        # Calculate how much the robot needs to turn
        turn_angle = target_heading - current_heading
        
        # Normalize the turn angle to be between -180 and 180 degrees
        if turn_angle > 180:
            turn_angle -= 360
        elif turn_angle < -180:
            turn_angle += 360
        
        print(f"Moving to ({target_x}, {target_y})")
        print(f"  - Turn angle: {turn_angle}°")
        print(f"  - Distance: {distance} mm")
        
        # Execute the turn
        robot.turn(turn_angle)
        
        # Execute the straight movement
        robot.straight(distance)
        
        # Update current position and heading
        current_x, current_y = target_x, target_y
        current_heading = target_heading
        
        # Optional: Add a small delay between movements
        ev3.speaker.beep()  # Beep to indicate reached target
    
    print("Navigation completed")

if __name__ == "__main__":

  
    robot_pos = (0, 0)
    target_points = [(300, 400), (700, 700), (500, 100), (200, 300)]
    
    print(f"Starting position: {robot_pos}")
    print(f"Target points: {target_points}")
    
    sorted_points = sort_proximity(robot_pos, target_points)
    print(f"Sorted points: {sorted_points}")
