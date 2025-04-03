#!/usr/bin/env python3

from time import sleep
from Robot import Robot
from Pathfinding import sort_proximity, move_robot

# =====================
# TEST DRIVE PATH
# =====================
def test_drive1():
    robot = Robot()

    target_points = [(110, 100), (200, 50), (150, 200)]
    obstacles = [(50, 50)]
    
    
    sorted_points = sort_proximity(robot.get_position(), target_points)
    print("Sorted points: ")
    print(str(sorted_points))

    move_robot(robot, target_points, obstacles)

# Run the test drive
test_drive1()