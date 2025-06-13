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
img_blob = None
cv2.namedWindow("Path Planner Visualization", cv2.WINDOW_NORMAL)
ROBOT_RADIUS = 30

def _mouse_callback(event, x, y, flags, param):
        """
        Mouse callback function to handle clicks on the grid.
        It allows the user to select start and end points for path planning.
        """
        global start, end, has_generated_path, ROBOT_RADIUS, img_blob
        if event == cv2.EVENT_LBUTTONDOWN:
            if start is None:
                start = (x, y)
                cv2.circle(img_blob, start, ROBOT_RADIUS, (0, 255, 0), -1)  # Draw start point
                cv2.imshow("Path Planner Visualization", img_blob)

            elif end is None:
                end = (x, y)
                cv2.circle(img_blob, end, ROBOT_RADIUS, (0, 0, 255), -1)  # Draw end point
                cv2.imshow("Path Planner Visualization", img_blob)

        if event == cv2.EVENT_RBUTTONDOWN:
            # Reset start and end points on right click
            start = None
            end = None
            has_generated_path = False

def demo_astar2(img_path="AI/images/image_432.jpg"):
    """
    Demo function to visualize the path planning on a course.
    """
    global start, end, has_generated_path, img_blob, ROBOT_RADIUS
    model = AIModel("ball_detect/v8/weights/best.pt")  # Load your YOLO model

    course = model.generate_course(img_path)  # Predict on an image
    course_viz = CourseVisualizer(draw_boxes=True, draw_labels=True, draw_masks=False)

    path_planner = PathPlanner(strategy=AStarStrategyOptimized(obj_radius=ROBOT_RADIUS))  # Using A* strategy with object radius of 2
    path_planner_viz = PathPlannerVisualizer()
    
    cv2.setMouseCallback("Path Planner Visualization", _mouse_callback)

    cv2.imshow("Path Planner Visualization", img_blob)

    while True:
        if start is not None and end is not None and not has_generated_path:
            path = path_planner.generate_path(start, end, grid)
            viz.draw_path(path)
            has_generated_path = True
            cv2.imshow("Path Planner Visualization", viz.img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            # Reset the visualization
            start = None
            end = None
            has_generated_path = False
            img_blob = viz.img.copy()
            cv2.imshow("Path Planner Visualization", img_blob)

if __name__ == "__main__":
    demo_astar2("AI/images/image_129.jpg")