from AImodel import AIModel
from Course import Course, CourseObject, CourseVisualizer
import cv2

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

if __name__ == "__main__":
    demo_course_visualize_maks()
