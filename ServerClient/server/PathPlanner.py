# import everything from Course
from Course import *
from AImodel import *

import cv2

import numpy as np
import math

'''
    PathPlanner is an object that convert the info from Course to a grid based 0,1 grid.

    It is used to plan the path for the robot to follow. It uses the strategy pattern 
    to allow different path planning strategies to be used.
    For now, it only has one strategy: A* algorithm.

    It can be used to find the shortest path from the any point, to any other point. 
    It account for the radius of the object (typically the robot) that is moving on the grid.
'''
class PathPlanner:
    # names: ['free', 'orange', 'white', 'egg', 'cross', 'robot', 'small_goal', 'big_goal', 'wall']
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

    def __init__(self):
        pass

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
        pass

    def display_grid(self, grid):
        """
        Display the grid using OpenCV.
        """
        width = len(grid[0])
        height = len(grid)
        img = np.zeros((height, width, 3), dtype=np.uint8)

        for y in range(height):
            for x in range(width):
                obj_num = grid[y, x]
                if obj_num in self.OBJECT_COLORS:
                    img[y, x] = self.OBJECT_COLORS[obj_num]

        cv2.imshow("Grid Visualization", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def demo_path_planner_visualization():
    """
    Demo function to visualize the path planning on a course.
    """
    model = AIModel("ball_detect/v8/weights/best.pt")  # Load your YOLO model
    course = model.generate_course("AI/images/image_432.jpg")  # Predict on an image

    path_planner = PathPlanner()
    grid = path_planner.generate_grid(course)

    viz = PathPlannerVisualizer()
    viz.display_grid(grid)

if __name__ == "__main__":
    demo_path_planner_visualization()