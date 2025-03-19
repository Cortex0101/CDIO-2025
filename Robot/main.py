#!/usr/bin/env python3

from ev3dev2.motor import LargeMotor, OUTPUT_A, OUTPUT_B, MoveTank
from time import sleep
from Robot import Robot
from Pathfinding import sort_proximity, move_robot

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