import cv2
from PathPlanner import *
from Course import *
from AImodel import *


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

    viz = PathPlannerVisualizer()
    img = viz.draw_grid_objects(grid)
    cv2.imshow("Path Planner Visualization", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    path = path_planner.find_path((120, 120), (350, 250), grid)
    img = viz.draw_path(img, path)
    cv2.imshow("Path on Grid", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
if __name__ == "__main__":
   demo_astar()