# import everything from Course
from Course import *
from AImodel import *

import cv2

import numpy as np
import math
import heapq

class AStarStrategy:
    '''
    A* pathfinding algorithm implementation with object radius.
    Ensures the path stays at least OBJ_RADIUS away from obstacles.
    '''

    def __init__(self, obj_radius=1):
        # Radius of the object (in grid cells)
        self.OBJ_RADIUS = obj_radius

    def _heuristic(self, a, b):
        # Using Euclidean distance as heuristic
        return np.linalg.norm(np.array(a) - np.array(b))

    def _in_bounds(self, point, grid_shape):
        x, y = point
        return 0 <= x < grid_shape[1] and 0 <= y < grid_shape[0]

    def _is_passable(self, point, grid):
        # Check surrounding cells within OBJ_RADIUS for obstacles
        x, y = point
        r = self.OBJ_RADIUS
        y_min = max(0, y - r)
        y_max = min(grid.shape[0] - 1, y + r)
        x_min = max(0, x - r)
        x_max = min(grid.shape[1] - 1, x + r)
        # If any cell in the square neighborhood is non-zero (obstacle), not passable
        neighborhood = grid[y_min:y_max+1, x_min:x_max+1]
        return np.all(neighborhood == 0)

    def find_path(self, start, end, grid):
        '''
        Find path from start to end using A*.

        start, end: (x, y)
        grid: 2D numpy array where 0=free, >0=obstacle
        '''
        # Priority queue: elements are (f_score, count, node)
        start = (int(start[0]), int(start[1]))
        end = (int(end[0]), int(end[1]))
        open_set = []
        heapq.heappush(open_set, (0 + self._heuristic(start, end), 0, start))
        came_from = {}

        g_score = { start: 0 }
        f_score = { start: self._heuristic(start, end) }

        visited = set()
        counter = 1

        # 8-directional moves: (dx, dy) and their costs
        neighbors = [
            (+1,  0, 1.0), (-1,  0, 1.0), (0, +1, 1.0), (0, -1, 1.0),
            (+1, +1, np.sqrt(2)), (+1, -1, np.sqrt(2)), (-1, +1, np.sqrt(2)), (-1, -1, np.sqrt(2))
        ]

        while open_set:
            _, _, current = heapq.heappop(open_set)

            if current == end:
                # reconstruct path
                path = []
                node = end
                while node in came_from:
                    path.append(node)
                    node = came_from[node]
                path.append(start)
                return path[::-1]

            visited.add(current)

            for dx, dy, cost in neighbors:
                neighbor = (current[0] + dx, current[1] + dy)

                if not self._in_bounds(neighbor, grid.shape):
                    continue
                if not self._is_passable(neighbor, grid):
                    continue

                tentative_g = g_score[current] + cost
                if neighbor in g_score and tentative_g >= g_score[neighbor]:
                    continue

                # This path is better
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + self._heuristic(neighbor, end)

                if neighbor not in visited:
                    heapq.heappush(open_set, (f_score[neighbor], counter, neighbor))
                    counter += 1

        # No path found
        return []

class AStarStrategyOptimized:
    '''
    A* pathfinding algorithm implementation with object radius.
    Optimized by pre-inflating obstacle grid and using array-based scores.
    '''

    def __init__(self, obj_radius=1):
        # Radius of the object (in grid cells)
        self.OBJ_RADIUS = obj_radius

    def set_object_radius(self, radius):
        '''
        Set the radius of the object for pathfinding.
        This is used to ensure the path stays at least OBJ_RADIUS away from obstacles.
        '''
        self.OBJ_RADIUS = radius

    def _heuristic(self, a, b):
        # Octile distance for 8-directional grid
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])
        return (dx + dy) + (np.sqrt(2) - 2) * min(dx, dy)

    def find_path(self, start, end, grid, exlude_obstacle_types=[0, 5]): #wall, robot, yellow, green
        '''
        Find path from start to end using A*.

        start, end: (x, y)
        grid: 2D numpy array where 0=free, >0=obstacle
        '''
        start = (int(start[0]), int(start[1]))
        end = (int(end[0]), int(end[1]))
        h, w = grid.shape
        # 1) Pre-inflate obstacles by OBJ_RADIUS
        #obstacles = (grid != 0).astype(np.uint8)
        obstacles = np.isin(grid, exlude_obstacle_types, invert=True).astype(np.uint8)
        kernel_size = 2 * self.OBJ_RADIUS + 1
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        inflated = cv2.dilate(obstacles, kernel)
        passable = (inflated == 0)

        # 2) Initialize score arrays and visited mask
        inf = float('inf')
        g_score = np.full((h, w), inf, dtype=np.float32)
        f_score = np.full((h, w), inf, dtype=np.float32)
        visited = np.zeros((h, w), dtype=bool)

        # 3) Open set (min-heap) with tie-breaker counter
        open_set = []
        counter = 0
        sx, sy = start
        ex, ey = end
        g_score[sy, sx] = 0.0
        f_score[sy, sx] = self._heuristic(start, end)
        heapq.heappush(open_set, (f_score[sy, sx], counter, start))

        # 4) Directions and costs
        dirs = [
            (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
            (1, 1, np.sqrt(2)), (1, -1, np.sqrt(2)), (-1, 1, np.sqrt(2)), (-1, -1, np.sqrt(2))
        ]
        came_from = dict()

        # 5) Main search loop
        while open_set:
            _, _, (cx, cy) = heapq.heappop(open_set)
            if visited[cy, cx]:
                continue
            visited[cy, cx] = True
            if (cx, cy) == (ex, ey):
                # Reconstruct path
                path = []
                node = (ex, ey)
                while node != (sx, sy):
                    path.append(node)
                    node = came_from[node]
                path.append((sx, sy))
                return path[::-1]

            for dx, dy, cost in dirs:
                nx, ny = cx + dx, cy + dy
                if not (0 <= nx < w and 0 <= ny < h):
                    continue
                if not passable[ny, nx]:
                    continue
                if visited[ny, nx]:
                    continue

                tentative_g = g_score[cy, cx] + cost
                if tentative_g < g_score[ny, nx]:
                    g_score[ny, nx] = tentative_g
                    f_score[ny, nx] = tentative_g + self._heuristic((nx, ny), (ex, ey))
                    came_from[(nx, ny)] = (cx, cy)
                    counter += 1
                    heapq.heappush(open_set, (f_score[ny, nx], counter, (nx, ny)))

        # No path found
        print("No path found from start to end.")
        return []

'''
    PathPlanner is an object that convert the info from Course to a grid based 0,1 grid.

    It is used to plan the path for the robot to follow. It uses the strategy pattern 
    to allow different path planning strategies to be used.
    For now, it only has one strategy: A* algorithm.

    It can be used to find the shortest path from the any point, to any other point. 
    It account for the radius of the object (typically the robot) that is moving on the grid.
'''
class PathPlanner:
    OBJECT_NUMS = {
        'wall': 0,  # Free space is represented by 0
        'orange': 1,
        'white': 2,
        'egg': 3,
        'cross': 4,
        'robot': 5,
        'small_goal': 6,
        'big_goal': 7,
        'outside_course': 8,  # This is used for the outside course area
    }

    def __init__(self, strategy=None):
        self.strategy = strategy

    def _polygon_fill_points(self, pts):
        # pts: array of shape (N,2), dtype=int
        x_min, y_min = pts.min(axis=0)
        x_max, y_max = pts.max(axis=0)
        # build a list of edges [(x0,y0,x1,y1), …]
        edges = []
        N = len(pts)
        for i in range(N):
            x0, y0 = pts[i]
            x1, y1 = pts[(i+1) % N]
            # ignore horizontal edges or swap to ensure y0 < y1
            if y0 == y1:
                continue
            if y0 > y1:
                x0, y0, x1, y1 = x1, y1, x0, y0
            edges.append((x0, y0, x1, y1))

        inside = []
        for y in range(y_min, y_max+1):
            x_intersects = []
            for x0, y0, x1, y1 in edges:
                if y0 <= y < y1:
                    # compute intersection
                    t = (y - y0) / (y1 - y0)
                    xi = x0 + t*(x1 - x0)
                    x_intersects.append(xi)
            x_intersects.sort()
            # fill between pairs
            for i in range(0, len(x_intersects), 2):
                x_start = int(np.ceil(x_intersects[i]))
                x_end   = int(np.floor(x_intersects[i+1]))
                inside.extend((x, y) for x in range(x_start, x_end+1))
        return inside

    def generate_grid(self, course: Course, excluded_objects: list = None, makeFloorEntireImage: bool = False):
        # fill grid with 8's (outside course area)
        grid = np.full((course.height, course.width), self.OBJECT_NUMS['outside_course'], dtype=np.uint8)
        
        # makeFloorEntireImage: for debugging purposes, if True, the floor will be the entire image excpet the outer most 20 pixels
        if makeFloorEntireImage:
            # fill the entire grid with walls except the outer most 20 pixels
            grid[20:course.height-20, 20:course.width-20] = self.OBJECT_NUMS['wall']
        else:
            floor = course.get_floor() # returns 'walls' which is the object representing the floor area
            if floor is not None:
                #grid[obj.y:obj.y + obj.height, obj.x:obj.x + obj.width] = self.OBJECT_NUMS['wall']
                y1 = (floor.bbox[1]).astype(int)
                y2 = (floor.bbox[3]).astype(int)
                x1 = (floor.bbox[0]).astype(int)
                x2 = (floor.bbox[2]).astype(int)
                grid[y1:y2, x1:x2] = self.OBJECT_NUMS['wall']

        for obj in course.objects:
            if obj.label == 'wall' or obj.label == 'green' or obj.label == 'yellow':
                continue
            
            should_skip = False
            for excluded in (excluded_objects or []):
                if obj is excluded:
                    should_skip = True
                    break
            
            if should_skip:
                continue
            else:
                pts = np.rint(obj.mask).astype(np.int32).reshape(-1, 2)
                coords_inside = self._polygon_fill_points(pts)
                for x, y in coords_inside:
                    if 0 <= x < course.width and 0 <= y < course.height:
                        grid[y, x] = self.OBJECT_NUMS[obj.label]

        return grid
    
    def set_object_radius(self, radius):
        """
        Set the radius of the object for pathfinding.
        This is used to ensure the path stays at least OBJ_RADIUS away from obstacles.
        
        Args:
            radius: int, radius of the object in grid cells
        """
        if self.strategy is not None:
            self.strategy.set_object_radius(radius)
        else:
            raise ValueError("No path planning strategy defined.")
    
    def find_path(self, start, end, grid):
        """
        Generate a path from start to end using the specified strategy.
        
        Args:
            start: tuple (x, y) starting point
            end: tuple (x, y) ending point
            grid: 2D numpy array representing the grid
        
        Returns:
            list of tuples representing the path from start to end
        """
        if self.strategy is None:
            raise ValueError("No path planning strategy defined.")
        
        return self.strategy.find_path(start, end, grid)
    
class PathPlannerVisualizer:
    OBJECT_COLORS = {
        1:     (255, 165,   0),   # orange
        2:      (255, 255, 255),   # white
        3:        (255, 105, 180),   # pinkish
        4:      (128,   0, 128),   # purple
        5:      (  0, 128, 255),   # sky blue
        6: (  0, 255, 128),   # mint green
        7:   (255,   0, 255),   # magenta
        0:       (128, 128,   0),   # olive
        8: (128, 128, 128)  # gray for outside course area
    }
    
    def __init__(self):
        self.start = None
        self.end = None

    def draw_grid_objects(self, grid):
        """
        Display the grid using OpenCV.
        """
        canvas = np.zeros((grid.shape[0], grid.shape[1], 3), dtype=np.uint8)
        for y in range(len(grid)):
            for x in range(len(grid[0])):
                obj_num = grid[y, x]
                if obj_num in self.OBJECT_COLORS:
                    canvas[y, x] = self.OBJECT_COLORS[obj_num]

        return canvas
    
    def draw_path(self, img, path):
        """
        Draw the path on the grid.
        
        Args:
            path: list of tuples representing the path to draw
        """
        canvas = img.copy()
        width, height = canvas.shape[1], canvas.shape[0]
        for x, y in path:
            if 0 <= x < width and 0 <= y < height:
                canvas[y, x] = (255, 0, 0)

        return canvas
    
    def draw_target_point(self, img, path_point):
        cv2.circle(img, path_point, 5, (0, 255, 0), -1)