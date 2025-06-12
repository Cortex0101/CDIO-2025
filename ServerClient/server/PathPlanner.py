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

    def generate_grid(self, course: Course):
        # fill grid with 8's (outside course area)
        grid = np.full((course.height, course.width), self.OBJECT_NUMS['outside_course'], dtype=np.uint8)
        
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

            pts = np.rint(obj.mask).astype(np.int32).reshape(-1, 2)

            coords_inside = self._polygon_fill_points(pts)
            for x, y in coords_inside:
                if 0 <= x < course.width and 0 <= y < course.height:
                    grid[y, x] = self.OBJECT_NUMS[obj.label]

        return grid
    
    def generate_path(self, start, end, grid):
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
    
    def __init__(self, grid, path_planner=None):
        self.path_planner = path_planner
        self.grid = grid
        self.width = len(grid[0])
        self.height = len(grid)
        self.img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.start = None
        self.end = None

    def draw_grid_objects(self):
        """
        Display the grid using OpenCV.
        """
        for y in range(self.height):
            for x in range(self.width):
                obj_num = self.grid[y, x]
                if obj_num in self.OBJECT_COLORS:
                    self.img[y, x] = self.OBJECT_COLORS[obj_num]

        return self.img
    
    def draw_path(self, path):
        """
        Draw the path on the grid.
        
        Args:
            path: list of tuples representing the path to draw
        """
        for x, y in path:
            if 0 <= x < self.width and 0 <= y < self.height:
                self.img[y, x] = (255, 0, 0)

def demo_path_planner_visualization():
    """
    Demo function to visualize the path planning on a course.
    """
    model = AIModel("ball_detect/v8/weights/best.pt")  # Load your YOLO model
    course = model.generate_course("AI/images/image_432.jpg")  # Predict on an image

    path_planner = PathPlanner(strategy=None)
    grid = path_planner.generate_grid(course)

    viz = PathPlannerVisualizer(grid)
    viz.draw_grid_objects()
    
    cv2.imshow("Path Planner Visualization", viz.img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def demo_astar():
    """
    Demo function to visualize the path planning on a course.
    """
    model = AIModel("ball_detect/v8/weights/best.pt")  # Load your YOLO model
    course = model.generate_course("AI/images/image_432.jpg")  # Predict on an image

    path_planner = PathPlanner(strategy=AStarStrategy(obj_radius=2))  # Using A* strategy with object radius of 2
    grid = path_planner.generate_grid(course)

    viz = PathPlannerVisualizer(grid)
    img = viz.draw_grid_objects()
    cv2.imshow("Path Planner Visualization", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    path = path_planner.generate_path((120, 120), (350, 250), grid)
    viz.draw_path(path)
    cv2.imshow("Path on Grid", viz.img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

start = None
end = None
has_generated_path = False

def _mouse_callback(event, x, y, flags, param):
        """
        Mouse callback function to handle clicks on the grid.
        It allows the user to select start and end points for path planning.
        """
        global start, end, has_generated_path
        if event == cv2.EVENT_LBUTTONDOWN:
            if start is None:
                start = (x, y)

            elif end is None:
                end = (x, y)

        if event == cv2.EVENT_RBUTTONDOWN:
            # Reset start and end points on right click
            start = None
            end = None
            has_generated_path = False

def demo_astar2():
    """
    Demo function to visualize the path planning on a course.
    """
    global start, end, has_generated_path
    model = AIModel("ball_detect/v8/weights/best.pt")  # Load your YOLO model
    course = model.generate_course("AI/images/image_432.jpg")  # Predict on an image

    path_planner = PathPlanner(strategy=AStarStrategy(obj_radius=2))  # Using A* strategy with object radius of 2
    grid = path_planner.generate_grid(course)

    viz = PathPlannerVisualizer(grid, path_planner=path_planner)
    viz.draw_grid_objects()
    img = viz.img.copy()
    
    cv2.namedWindow("Path Planner Visualization", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Path Planner Visualization", _mouse_callback)

    print("Click to select start and end points for path planning.")
    print("Press 'q' to exit.")

    cv2.imshow("Path Planner Visualization", img)

    while True:
        if start is not None and end is not None and not has_generated_path:
            path = path_planner.generate_path(start, end, grid)
            viz.draw_path(path)
            has_generated_path = True
            cv2.imshow("Path Planner Visualization", viz.img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

if __name__ == "__main__":
    demo_astar2()