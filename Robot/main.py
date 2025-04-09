#!/usr/bin/env python3

from time import sleep
from Robot import Robot
from Pathfinding import sort_proximity, move_robot

# =====================
# TEST DRIVE PATH
# =====================
def test_drive1():
    robot = Robot()

    target_points = [(100, 100)]
    obstacles = [(50, 50)]
    
    move_robot(robot, target_points, obstacles)

# Run the test drive
test_drive1()