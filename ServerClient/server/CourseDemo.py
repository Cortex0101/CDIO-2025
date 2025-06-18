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

    visualizer = CourseVisualizer(draw_walls=False, draw_boxes=True)
    img = cv2.imread("AI/images/image_138.jpg")
    img = visualizer.draw(img, course)

    all_white_balls = course.get_white_balls()
    for ball in all_white_balls:
        random_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        img = visualizer.highlight_ball(img, ball, color=random_color)
        optimal_spot = course.get_optimal_ball_parking_spot(ball)
        if optimal_spot:
            print(f"Optimal parking spot for ball {random_color[0]}, {random_color[1]}, {random_color[2]}: {optimal_spot}")
            img = visualizer.highlight_point(img, optimal_spot, random_color)

    cv2.imshow("Nearest Ball Visualization", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def demo_visualize_balls_near_corners():
    """
    Demo function to visualize balls near corners in a Course.
    """
    model = AIModel("ball_detect/v8/weights/best.pt")  # Load your YOLO model
    course = model.generate_course("AI/images/image_170.jpg")  # Predict on an image

    visualizer = CourseVisualizer(draw_walls=False, draw_boxes=True)
    img = cv2.imread("AI/images/image_170.jpg")
    img = visualizer.draw(img, course)

    balls = course.get_white_balls()

    for ball in balls:
        if course.is_ball_near_corner(ball):
            color = (25, 75, 120)  # Color for balls near corners
            print(f"Ball near corner: {ball}")
            img = visualizer.highlight_ball(img, ball, color=color)
        elif course.is_ball_near_wall(ball, threshold=25):
            color = (120, 75, 25)
            print(f"Ball near wall: {ball}")
            img = visualizer.highlight_ball(img, ball, color=color)
        elif course.is_ball_near_cross(ball):
            color = (75, 120, 25)
            print(f"Ball near cross: {ball}")
            img = visualizer.highlight_ball(img, ball, color=color)

    nearest_goal = course.get_nearest_goal(course.get_robot().center)
    if nearest_goal:
        print(f"Nearest goal: {nearest_goal}")
        visualizer.highlight_point(img, nearest_goal.center, (0, 255, 0))  # Highlight nearest goal in green
    
    cv2.imshow("Balls Near Corners Visualization", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    demo_visualize_balls_near_corners()
