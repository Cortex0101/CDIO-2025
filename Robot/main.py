#!/usr/bin/env python3

from ev3dev2.motor import LargeMotor, OUTPUT_A, OUTPUT_B, MoveTank
from time import sleep

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
def test_drive():
    """
    Perform a test drive: move forward, turn, move back.
    """
    print("Starting test drive...")

    while True:

        move_forward(20, speed=50)  # Move forward 20 cm
        sleep(1)

        turn_right(90, speed=30)  # Turn right 90 degrees
        sleep(1)

        move_forward(10, speed=50)  # Move forward 10 cm
        sleep(1)

        turn_left(45, speed=30)  # Turn left 45 degrees
        sleep(1)

        move_backward(15, speed=50)  # Move backward 15 cm
        sleep(1)

    print("Test drive complete.")

# Run the test drive
test_drive()