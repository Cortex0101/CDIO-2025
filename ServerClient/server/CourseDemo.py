from AImodel import AIModel
from Course import Course, CourseObject, CourseVisualizer
import cv2
import random

def demo_generate_and_print_course_objects():
    """
    Demo function to generate a Course and print its objects.
    """
    model = AIModel("ball_detect/v8/weights/best.pt")  # Load your YOLO model
    course = model.generate_course("AI/images/image_432.jpg")  # Predict on an image
    print(course)  # Print detected objects
    for obj in course:
        print(obj)  # Print each CourseObject's details

def demo_visualize_course():
    """
    Demo function to visualize a Course on an image.
    """
    model = AIModel("ball_detect/v8/weights/best.pt")  # Load your YOLO model
    course = model.generate_course("AI/images/image_432.jpg")  # Predict on an image

    visualizer = CourseVisualizer()
    img = cv2.imread("AI/images/image_432.jpg")
    result_img = visualizer.draw(img, course)

    cv2.imshow("Course Visualization", result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def demo_course_visualize_maks():
    """
    Demo function to visualize a Course with masks.

    self,
                 draw_centers: bool = True,
                 draw_center_points: bool = True,
                 draw_walls:   bool = False,
                 draw_direction_markers: bool = False,
                 draw_masks:   bool = False,
                 draw_boxes:   bool = True,
                 draw_confidence: bool = False,
                 draw_labels: bool = True,
                 mask_alpha:   float = 0.1):
    """
    model = AIModel("ball_detect/v8/weights/best.pt")  # Load your YOLO model
    course = model.generate_course("AI/images/image_432.jpg")  # Predict on an image

    visualizer = CourseVisualizer(draw_boxes=True,
                                    draw_masks=True,
                                    draw_labels=True,
                                    draw_centers=True,
                                    mask_alpha=0.2)  # Adjust parameters as needed
    img = cv2.imread("AI/images/image_432.jpg")
    res = visualizer.draw(img, course)

    cv2.imshow("Course Visualization with Masks", res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def demo_visualize_nearest_ball():
    """
    Demo function to visualize the nearest ball in a Course.
    """
    model = AIModel("ball_detect/v8/weights/best.pt")  # Load your YOLO model
    course = model.generate_course("AI/images/image_138.jpg")  # Predict on an image

    visualizer = CourseVisualizer(draw_walls=True, draw_boxes=False)
    img = cv2.imread("AI/images/image_138.jpg")
    img = visualizer.draw(img, course)
    robot = course.get_robot()
    if not robot:
        print("No robot found in the course.")
        return
    
    # Assuming the robot has a 'center' attribute for its position
    robot_center = robot.center
    nearest_ball = course.get_nearest_ball(robot_center)  # Example point
    
    if nearest_ball:
        img = visualizer.highlight_ball(img, nearest_ball)

    all_white_balls = course.get_white_balls()
    for ball in all_white_balls:
        random_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        img = visualizer.highlight_ball(img, ball, color=random_color)
        optimal_spot = course.get_optimal_ball_parking_spot(ball, robot)
        if optimal_spot:
            print(f"Optimal parking spot for ball {random_color[0]}, {random_color[1]}, {random_color[2]}: {optimal_spot}")
            img = visualizer.highlight_point(img, optimal_spot, random_color)

    cv2.imshow("Nearest Ball Visualization", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    demo_visualize_nearest_ball()
