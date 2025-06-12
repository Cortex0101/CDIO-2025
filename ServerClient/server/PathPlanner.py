# import everything from Course
from Course import *
from AImodel import *

import cv2

'''
    PathPlanner is an object that convert the info from Course to a grid based 0,1 grid.

    It is used to plan the path for the robot to follow. It uses the strategy pattern 
    to allow different path planning strategies to be used.
    For now, it only has one strategy: A* algorithm.

    It can be used to find the shortest path from the any point, to any other point. 
    It account for the radius of the object (typically the robot) that is moving on the grid.
'''
class PathPlanner:
    def __init__(self, course: Course, 
                 robot_radius: float = 5.5,
                 grid_scale: float = 1.0):
        self.course = course
        self.robot_radius = robot_radius
        self.grid_scale = grid_scale

    def generate_grid(self, obstacle_padding: int = 0):
        """
        Generate a grid representation of the course.
        The grid is a 2D list where each cell is either 0 (free) or 1 (obstacle).
        The size of the grid is determined by the course dimensions and the grid scale.

        Args:
            obstacle_padding (int): Extra padding around obstacles to ensure the robot can navigate around them.

        Returns:
            list: A 2D list representing the grid, where 
            0 is free space
            And 1 - 9 is:
            ['orange', 'white', 'egg', 'cross', 'small_goal', 'big_goal']
        """
        OBJECT_NUMS = {
            'orange': 1,
            'white': 2,
            'egg': 3,
            'cross': 4,
            'small_goal': 5,
            'big_goal': 6
        }

        width = 640
        height = 480
        grid = [[0 for _ in range(width)] for _ in range(height)]

        for obj in self.course.objects:
                if obj.label == 'wall' or obj.label == 'robot':
                    continue # Walls and the robot itself are not considered obstacles in this context

                x, y = obj.bbox[0] / self.grid_scale, obj.bbox[1] / self.grid_scale
                radius = (obj.bbox[2] - obj.bbox[0]) / 2 / self.grid_scale + obstacle_padding
                x_start = max(0, int(x - radius))
                x_end = min(width, int(x + radius))
                y_start = max(0, int(y - radius))
                y_end = min(height, int(y + radius))

                for i in range(y_start, y_end):
                    for j in range(x_start, x_end):
                        if (i - y) ** 2 + (j - x) ** 2 <= radius ** 2:
                            grid[i][j] = OBJECT_NUMS.get(obj.label, 0)  # Use the object number or 0 if not found

        return grid
    
class PathPlannerVisualizer:
    OBJECT_COLORS = {
        "orange":     (255, 165,   0),   # orange
        "white":      (255, 255, 255),   # white
        "egg":        (255, 105, 180),   # pinkish
        "cross":      (128,   0, 128),   # purple
        "robot":      (  0, 128, 255),   # sky blue
        "small_goal": (  0, 255, 128),   # mint green
        "big_goal":   (255,   0, 255),   # magenta
        "wall":       (128, 128,   0),   # olive
    }
    
    def __init__(self, course: Course):
        self.course = course

    def display_grid(self, grid):
        """
        Draw the grid on a new image with 
        """
        # new image of size 640x480
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        height, width = img.shape[:2]
        cell_height = height / len(grid)
        cell_width = width / len(grid[0])

        for i in range(len(grid)):
            for j in range(len(grid[i])):
                cell_value = grid[i][j]
                if cell_value == 0:
                    continue

                # Get the color for the object
                color = self.OBJECT_COLORS.get(self.course.objects[cell_value - 1].label, (0, 0, 0))
                # Calculate the rectangle coordinates
                x1 = int(j * cell_width)
                y1 = int(i * cell_height)
                x2 = int((j + 1) * cell_width)
                y2 = int((i + 1) * cell_height)
                # Draw the rectangle
                cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
        # Resize the image to fit the screen
        img = cv2.resize(img, (800, 600), interpolation=cv2.INTER_AREA)

        # Show the image
        cv2.imshow("Grid Visualization", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

