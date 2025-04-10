#!/usr/bin/env python3

from time import sleep
from Robot import Robot
from Pathfinding import sort_proximity, move_robot

# =====================
# TEST DRIVE PATH
# =====================
robot = Robot()

def test_drive1():

    target_points = [(160,100)]
    obstacles = [(90,60)]
    
    move_robot(robot, target_points, obstacles)

# Run the test drive
test_drive1()

sleep(10)

target_points = [(20,100)]
move_robot(robot, target_points)