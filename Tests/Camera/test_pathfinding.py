import pytest
from Camera.Pathfinding import *

def test_calculate_distance():
    # Test case where both points are at the same location
    point1 = (0, 0)
    point2 = (0, 0)
    
    # Calculate the distance between the two points
    distance = calculate_distance(point1, point2)
    
    # Assert that the distance is 0
    assert distance == 0, f"Expected distance 0, but got {distance}"

    # Test case for points at different locations
    point1 = (0, 0)
    point2 = (3, 4)
    
    # Calculate the distance between the two points
    distance = calculate_distance(point1, point2)
    
    # Assert that the distance is correct (5 from Pythagorean theorem)
    assert distance == 5, f"Expected distance 5, but got {distance}"

def test_sort_proximity():
    robot_position = (0, 0)
    points = [(2, 2), (1, 2), (0, 2)]
    
    # Sort points based on proximity to the robot's position
    sorted_points = sort_proximity(robot_position, points)
    
    # Assert that the sorted points are in the correct order based on distance
    assert sorted_points == [(0, 2), (1, 2), (2, 2)], f"Expected sorted points [(0, 2), (1, 2), (2, 2)], but got {sorted_points}"

def test_line_intersects_obstacle():
    start = (0, 0)
    end = (10, 10)
    obstacle = (5, 5)
    obstacle_radius = 2

    # Test case where the line intersects the obstacle
    result = line_intersects_obstacle(start, end, obstacle, obstacle_radius)
    
    # Assert that the line intersects the obstacle
    assert result == True, f"Expected True, but got {result}"

    # Test case where the line does not intersect the obstacle
    obstacle = (2, 8)
    
    result = line_intersects_obstacle(start, end, obstacle, obstacle_radius)
    
    # Assert that the line does not intersect the obstacle
    assert result == False, f"Expected False, but got {result}"

def test_avoid_obstacles():
    startpoint = (0, 0)
    endpoint = (10, 10)
    obstacles = [(5, 5), (8, 8)]  # Obstacles in the path
    obstacle_radius = 2
    
    # Call the function to avoid obstacles
    path = avoid_obstacles(startpoint, endpoint, obstacles, obstacle_radius)
    
    # Assert that the returned path contains waypoints and ends at the endpoint
    assert path[-1] == endpoint, f"Expected endpoint {endpoint}, but got {path[-1]}"
    assert len(path) > 1, f"Expected more than one waypoint, but got {len(path)}"

def test_calibrate_angle():
    # Test case where current angle matches camera input
    current_angle = 30
    camera_input = 30
    
    # Call the function to calibrate the angle
    calibration = calibrate_angle(current_angle, camera_input)
    
    # Assert that no correction is needed
    assert calibration == 0.0, f"Expected calibration 0.0, but got {calibration}"

    # Test case where current angle does not match camera input
    current_angle = 45
    camera_input = 90
    
    # Call the function to calibrate the angle
    calibration = calibrate_angle(current_angle, camera_input)
    
    # Assert that the calibration equals the difference
    assert calibration == 45.0, f"Expected calibration 45.0, but got {calibration}"