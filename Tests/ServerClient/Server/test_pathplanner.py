import numpy as np
import pytest

from ServerClient.server.PathPlanner import AStarStrategyOptimized

@pytest.fixture
def astar():
    return AStarStrategyOptimized(obj_radius=1)

def test_simple_path_no_obstacles(astar):
    grid = np.zeros((10, 10), dtype=np.uint8)
    start = (0, 0)
    end = (9, 9)
    path = astar.find_path(start, end, grid)
    assert len(path) > 0
    assert path[0] == start
    assert path[-1] == end

def test_no_path_due_to_obstacle(astar):
    grid = np.zeros((5, 5), dtype=np.uint8)
    grid[2, :] = 1  # Obstacle row
    start = (0, 0)
    end = (4, 4)
    path = astar.find_path(start, end, grid)
    assert path == []

def test_start_or_end_out_of_bounds(astar):
    grid = np.zeros((5, 5), dtype=np.uint8)
    start = (-1, 0)
    end = (4, 4)
    path = astar.find_path(start, end, grid)
    assert path == []
    start = (0, 0)
    end = (5, 5)
    path = astar.find_path(start, end, grid)
    assert path == []

def test_start_or_end_not_passable(astar):
    grid = np.zeros((5, 5), dtype=np.uint8)
    grid[0, 0] = 1  # Obstacle at start
    grid[4, 4] = 1  # Obstacle at end
    start = (0, 0)
    end = (4, 4)
    path = astar.find_path(start, end, grid)
    assert path == []

def test_object_radius_blocks_path(astar):
    grid = np.zeros((7, 7), dtype=np.uint8)
    grid[3, :] = 1  # Obstacle row
    astar.set_object_radius(2)
    start = (0, 0)
    end = (6, 6)
    path = astar.find_path(start, end, grid)
    assert path == []

def test_object_radius_allows_path(astar):
    grid = np.zeros((7, 7), dtype=np.uint8)
    grid[3, 1:6] = 1  # Obstacle with gap at (3,0) and (3,6)
    astar.set_object_radius(1)
    start = (0, 0)
    end = (6, 6)
    path = astar.find_path(start, end, grid)
    assert len(path) > 0
    assert path[0] == start
    assert path[-1] == end

def test_exclude_obstacle_types(astar):
    grid = np.zeros((5, 5), dtype=np.uint8)
    grid[2, 2] = 5  # robot, which is excluded by default
    start = (0, 0)
    end = (4, 4)
    path = astar.find_path(start, end, grid)
    assert len(path) > 0
    assert path[0] == start
    assert path[-1] == end
