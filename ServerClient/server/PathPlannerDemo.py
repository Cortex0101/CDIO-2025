from PathPlanner import *
from Course import *
from AImodel import *

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