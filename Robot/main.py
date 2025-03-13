#!/usr/bin/env python3

from ev3dev2.motor import LargeMotor, OUTPUT_A, OUTPUT_B, MoveTank
from time import sleep
from Robot import Robot
from Pathfinding import sort_proximity, move_robot

# Initialize motors
left_motor = LargeMotor(OUTPUT_A)
right_motor = LargeMotor(OUTPUT_B)
tank_drive = MoveTank(OUTPUT_A, OUTPUT_B)

# Constants
WHEEL_DIAMETER = 4.2  # cm (EV3 Medium Motor wheel size)
WHEEL_CIRCUMFERENCE = 3.1416 * WHEEL_DIAMETER  # cm per full wheel rotation
AXLE_TRACK = 12  # cm (distance between left and right wheels)

def move_forward(distance_cm, speed=50):
    """
    Move forward a certain distance in cm.
    """
    rotations = distance_cm / WHEEL_CIRCUMFERENCE  # Convert distance to motor rotations
    tank_drive.on_for_rotations(speed, speed, rotations)

def move_backward(distance_cm, speed=50):
    """
    Move backward a certain distance in cm.
    """
    rotations = distance_cm / WHEEL_CIRCUMFERENCE
    tank_drive.on_for_rotations(-speed, -speed, rotations)

def turn_left(angle, speed=30):
    """
    Turn left by a certain angle (in degrees).
    """
    turn_distance = (angle / 360) * (3.1416 * AXLE_TRACK)  # Calculate arc length
    rotations = turn_distance / WHEEL_CIRCUMFERENCE
    tank_drive.on_for_rotations(-speed, speed, rotations)  # Left wheel moves backward, right forward

def turn_right(angle, speed=30):
    """
    Turn right by a certain angle (in degrees).
    """
    turn_distance = (angle / 360) * (3.1416 * AXLE_TRACK)
    rotations = turn_distance / WHEEL_CIRCUMFERENCE
    tank_drive.on_for_rotations(speed, -speed, rotations)  # Right wheel moves backward, left forward

# =====================
# TEST DRIVE PATH
# =====================
def test_drive1():
    robot = Robot()

    # Robot should move straight, turn right, move straight, then turn around and move back
    target_points = [(0, 40), (40, 40), (80, 80)]
    
    #print("Starting position: x=" + robot.get_position(
    #print("Target points: " + target_points)
    
    sorted_points = sort_proximity(robot.get_position(), target_points)
    print("Sorted points: ")
    print(str(sorted_points))

    move_robot(robot, target_points)

# Run the test drive
test_drive1()