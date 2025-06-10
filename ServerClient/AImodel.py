from ultralytics import YOLO
import cv2
import numpy as np
from scipy.interpolate import splprep, splev

class CourseObject:
    def __init__(self, name, center, bbox, confidence, mask):
        self.name = name # e.g. "robot", "white_ball", "orange_ball", "egg", "small_goal", "large_goal", "wall", "cross"
        self.center = center  # (x, y)
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.mask = mask  # Polygon mask if available, otherwise None
        self.confidence = confidence

    def __repr__(self):
        return f"{self.name} at {self.center} with bbox {self.bbox} and confidence {self.confidence:.2f}"

# contains the all the objects that are detected in the course, such as the robot, balls, goals, walls, etc.
# contains utilities such as a counter
class Course:
    def __init__(self):
        self.objects = []  # List of CourseObject instances
        self.object_count = {}

    @staticmethod
    def stream_to_model_results_to_course_objects(model_results):
        """
        Converts model results to a list of CourseObject instances.
        
        Args:
            model_results: The results from the YOLO model prediction.
        
        Returns:
            List of CourseObject instances.
        """
        objects = []
        for box in model_results.boxes:
            cls_id = int(box.cls[0].item())
            cls_name = model_results.names[cls_id]
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            confidence = float(box.conf[0].item())
            mask = box.masks.xy if hasattr(box, 'masks') else None
            
            obj = CourseObject(name=cls_name, center=(cx, cy), bbox=[x1, y1, x2, y2], confidence=confidence, mask=mask)
            objects.append(obj)
        
        return objects

    def add_object(self, obj: CourseObject):
        self.objects.append(obj)
        if obj.name in self.object_count:
            self.object_count[obj.name] += 1
        else:
            self.object_count[obj.name] = 1

    def get_objects_by_name(self, name):
        return [obj for obj in self.objects if obj.name == name]

    def __repr__(self):
        return f"Course with {len(self.objects)} objects: {self.object_count}"

class AIModel:
    def __init__(self):
        self.model = YOLO("ball_detect/v7/weights/best.pt")
        
        self.previous_results = []

        self.current_frame = None
        self.current_results = None
        self.current_processed_drawn_frame = None
        
        self.current_course = Course()
        self.previous_courses = []  # List of Course instances to keep track of previous courses

        self.SHOW_BOXES = False
        self.SHOW_MASKS = False
        self.SHOW_CONFIDENCE = True
        self.SHOW_LABELS = True
        self.SHOW_CENTER = True

        self.COLORS = {
            "egg": (255, 0, 0),      # Red
            "robot": (0, 255, 0),    # Green
            "white_ball": (255, 255, 255),  # White
            "orange_ball": (0, 165, 255),   # Orange
            "small_goal": (255, 0, 0),  # Blue
            "large_goal": (255, 255, 0), # Yellow
            "wall": (128, 128, 128),    # Gray
            "cross": (0, 0, 145),       # Dark Blue
        }

        self.excluded_classes = [
            # empty by default, can be filled with class names to exclude
        ]

    def set_options(self, show_boxes=True, show_masks=False, show_confidence=True, show_labels=True, show_center=True):
        """
        Set options for drawing results.
        
        Args:
            show_boxes (bool): Whether to show bounding boxes.
            show_masks (bool): Whether to show masks.
            show_confidence (bool): Whether to show confidence scores.
            show_labels (bool): Whether to show labels.
            show_center (bool): Whether to show center points.
        """
        self.SHOW_BOXES = show_boxes
        self.SHOW_MASKS = show_masks
        self.SHOW_CONFIDENCE = show_confidence
        self.SHOW_LABELS = show_labels
        self.SHOW_CENTER = show_center

    def set_excluded_classes(self, classes):
        """
        Set classes to exclude from detection results.
        
        Args:
            classes (list): List of class names to exclude.
        """
        self.excluded_classes = classes

    def process_frame(self, frame):
        self.current_results = self.model.predict(source=frame, conf=0.3, iou=0.5)[0]  # Get the first result
        if not self.current_results:
            print("No results found.")
            return None
        self.current_frame = frame
        self.current_processed_drawn_frame = frame.copy()  # Create a copy for drawing
        self.previous_results.append(self.current_results)
        if len(self.previous_results) > 10:  # Keep only the last 10 results
            self.previous_results.pop(0)

        self.current_course.objects = Course.stream_to_model_results_to_course_objects(self.current_results)
        self.previous_courses.append(self.current_course)
        if len(self.previous_courses) > 10:
            self.previous_courses.pop(0)

        self.draw_results()  # Draw results on the current processed frame

    def draw_results(self):
        for object in self.current_course.objects:
            if object.name in self.excluded_classes:
                continue
            
            if self.SHOW_BOXES:
                cv2.rectangle(self.current_processed_drawn_frame, (object.bbox[0], object.bbox[1]), (object.bbox[2], object.bbox[3]), self.COLORS.get(object.name, (0, 255, 0)), 2)
            if self.SHOW_MASKS and object.mask is not None:
                pts = np.array(object.mask, dtype=np.int32)
                cv2.fillPoly(self.current_processed_drawn_frame, [pts], self.COLORS.get(object.name, (0, 0, 255)))
            if self.SHOW_LABELS:
                label = f"{object.name} {object.confidence:.2f}" if self.SHOW_CONFIDENCE else object.name
                cv2.putText(self.current_processed_drawn_frame, label, (object.bbox[0], object.bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            if self.SHOW_CENTER:
                cx = int((object.bbox[0] + object.bbox[2]) / 2)
                cy = int((object.bbox[1] + object.bbox[3]) / 2)
                cv2.circle(self.current_processed_drawn_frame, (cx, cy), 5, (0, 0, 255), -1)
                cv2.putText(self.current_processed_drawn_frame, f"({cx},{cy})", (cx + 5, cy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def find_closest_ball(self, ball_type: str = "white") -> dict | None:
        """
        Finds the closest ball of the specified color (or if "either", any ball) in the current results.

        Args:
            ball_color: Color of the ball to find (e.g., "white", "orange" or "either").

        Returns:
            A dict with 'class', 'confidence', 'bbox', 'centroid' if found, else None.
        """
        if self.current_results is None or self.current_results.boxes is None:
            return None

        closest_ball = None
        min_distance = float('inf')

        for box in self.current_results.boxes:
            cls_id = int(box.cls[0].item())
            cls_name = self.current_results.names[cls_id]
            if ball_type != "either" and cls_name != ball_type:
                continue

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            distance = np.sqrt(cx**2 + cy**2)

            if distance < min_distance:
                min_distance = distance
                closest_ball = {
                    'class': cls_name,
                    'confidence': float(box.conf[0].item()),
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'center': (float(cx), float(cy))
                }

        return closest_ball
    
    def highlight_ball(self, ball, include_label=True): 
        """
        Highlights the closest ball in the current processed frame by drawing a rectangle and label on that frame.
        This method modifies the `self.current_processed_drawn_frame` attribute.

        Args:
            ball: Dict with 'class', 'confidence', 'bbox', 'centroid'.
        """
        if ball is None:
            return

        x1, y1, x2, y2 = ball['bbox']
        cv2.rectangle(self.current_processed_drawn_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cx, cy = ball['center']
        cv2.circle(self.current_processed_drawn_frame, (int(cx), int(cy)), 5, (0, 0, 255), -1)

        if include_label:
            cv2.putText(self.current_processed_drawn_frame, f"{ball['class']} {ball['confidence']:.2f}",
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
    def do_boxes_overlap(self, box1, box2):
        """
        Check if two bounding boxes overlap.
        
        Args:
            box1: First bounding box as [x1, y1, x2, y2].
            box2: Second bounding box as [x1, y1, x2, y2].
        
        Returns:
            bool: True if boxes overlap, False otherwise.
        """
        x1_max = max(box1[0], box2[0])
        y1_max = max(box1[1], box2[1])
        x2_min = min(box1[2], box2[2])
        y2_min = min(box1[3], box2[3])
        
        return (x1_max < x2_min) and (y1_max < y2_min)
    
    def find_overlapping_boxes(self, exclude_classes=None):
        """
        Find all boxes that overlap with any box

        Args:
            exclude_classes (list): List of class names to exclude from overlap detection. If None, no classes are excluded.
        
        Returns:
            list: List of box pairs that overlap, each represented as a list of coordinates [x1, y1, x2, y2].
        """
        if self.current_results is None or self.current_results.boxes is None:
            return []

        boxes = self.current_results.boxes.xyxy.cpu().numpy().astype(int)
        overlapping_boxes = []

        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                box1 = boxes[i]
                box2 = boxes[j]

                if exclude_classes:
                    cls1 = self.current_results.names[int(self.current_results.boxes.cls[i].item())]
                    cls2 = self.current_results.names[int(self.current_results.boxes.cls[j].item())]
                    if cls1 in exclude_classes or cls2 in exclude_classes:
                        continue

                if self.do_boxes_overlap(box1, box2):
                    overlapping_boxes.append((box1, box2))

        return overlapping_boxes
    
    def highlight_overlapping_boxes(self, exclude_classes=None):
        """
        Highlights all overlapping boxes in the current processed frame.
        This method modifies the `self.current_processed_drawn_frame` attribute.

        Draws a line from the center of one box to the center of the other.
        """
        overlapping_boxes = self.find_overlapping_boxes(exclude_classes=exclude_classes)
        
        for boxes in overlapping_boxes:
            box1, box2 = boxes
            x1_1, y1_1, x2_1, y2_1 = box1
            x1_2, y1_2, x2_2, y2_2 = box2
            
            # Calculate centers
            cx1 = int((x1_1 + x2_1) / 2)
            cy1 = int((y1_1 + y2_1) / 2)
            cx2 = int((x1_2 + x2_2) / 2)
            cy2 = int((y1_2 + y2_2) / 2)
            
            # Draw line between centers
            cv2.line(self.current_processed_drawn_frame, (cx1, cy1), (cx2, cy2), (255, 0, 0), 3)

    def get_objects(self):
        """
        Returns the current results of the model.

        But is returned as a dict like:
        {
            robot: {
                "center": (x, y),
                "bbox": [x1, y1, x2, y2],
                "confidence": float,
            },
            "white": [
                {
                    "center": (x, y),
                    "bbox": [x1, y1, x2, y2],
                    "confidence": float,
                },
                ...
            ],
            "orange": [
                {
                    "center": (x, y),
                    "bbox": [x1, y1, x2, y2],
                    "confidence": float,
                },
                ...
            ],
        }
        
        such that one can easily access the objects by their class name.
        """
        if self.current_results is None or self.current_results.boxes is None:
            return {}

        objects = {}
        for box in self.current_results.boxes:
            cls_id = int(box.cls[0].item())
            cls_name = self.current_results.names[cls_id]
            if cls_name in self.excluded_classes:
                continue

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            confidence = float(box.conf[0].item())

            if cls_name not in objects:
                objects[cls_name] = []

            objects[cls_name].append({
                'center': (float(cx), float(cy)),
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'confidence': confidence
            })

        return objects

    def plan_smooth_path(self, target_coord, obstacle_padding=5, max_curvature=0.05, num_waypoints=100):
        """
        Generate a smooth, obstacle-avoiding path from robot to target using bounding boxes.

        Args:
            robot_centroid: (x,y) robot center.
            target_centroid: (x,y) goal center.
            obstacle_bboxes: list of [x1,y1,x2,y2] for obstacles.
            robot_radius: radius of robot in pixels.
            obstacle_padding: extra padding around obstacles.
            max_curvature: smoothing factor (lower -> tighter turns).
            num_waypoints: number of sampled points.

        Returns:
            smooth_path: ndarray of shape (N,2) of x,y waypoints.
        """
        objects = self.get_objects()
        robot_centroid = objects['robot'][0]['center']
        target_centroid = target_coord  # Assuming target_coord is provided as (x, y)

        # get all bboxes of all objects that dont have the same center coordinate as the robot or target
        obstacle_bboxes = []
        for obj in objects.values():
            if isinstance(obj, list):
                for item in obj:
                    if item['center'] != robot_centroid and item['center'] != target_centroid:
                        obstacle_bboxes.append(item['bbox'])

        robot_radius = objects['robot'][0]['bbox'][2] - objects['robot'][0]['bbox'][0]  # width of robot bbox
        robot_radius /= 2.0  # use half of width as radius

        # 1. Sample straight-line path
        t = np.linspace(0, 1, num_waypoints)
        line = np.outer(1 - t, robot_centroid) + np.outer(t, target_centroid)

        # 2. Build obstacle circles (center, radius)
        circles = []
        for (x1, y1, x2, y2) in obstacle_bboxes:
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            # approximate radius as half of diagonal
            r = np.hypot(x2 - x1, y2 - y1) / 2.0
            circles.append({'center': np.array([cx, cy], dtype=float),
                            'radius': r + robot_radius + obstacle_padding})

        # 3. Apply repulsion to path points
        path = line.copy()
        for i in range(1, num_waypoints - 1):
            pt = path[i]
            for c in circles:
                vec = pt - c['center']
                dist = np.linalg.norm(vec)
                if dist < c['radius'] and dist > 1e-3:
                    # push point away
                    push = (c['radius'] - dist)
                    path[i] += (vec / dist) * push

        # 4. Interpolate smooth spline
        tck, u = splprep([path[:, 0], path[:, 1]], s=max_curvature * num_waypoints)
        u_new = np.linspace(0, 1, num_waypoints)
        x_new, y_new = splev(u_new, tck)
        smooth_path = np.vstack([x_new, y_new]).T
        return smooth_path
    
    def draw_path(self, path, color=(0, 255, 255), thickness=2):
        """
        Draws a polyline on the frame for the given path.

        Returns:
            frame: image with path overlay.
        """
        pts = np.array(path, dtype=np.int32)
        cv2.polylines(self.current_processed_drawn_frame, [pts], isClosed=False, color=color, thickness=thickness)

    def get_ball_overlapping_with_robot(self): 
        """
        Finds the ball that overlaps with the robot in the current processed frame.

        Returns:
            dict: Ball information if found, else None.
        """
        objects = self.get_objects()
        if 'robot' not in objects or 'white' not in objects:
            return None

        robot_bbox = objects['robot'][0]['bbox']
        for ball in objects['white']:
            ball_bbox = ball['bbox']
            if self.do_boxes_overlap(robot_bbox, ball_bbox):
                ball['centroid'] = ball['center']
                return ball

        return None

if __name__ == "__main__":
    model = AIModel()

    model.set_options(show_boxes=True, show_masks=False, show_confidence=False, show_labels=False, show_center=False)
    model.set_excluded_classes(['wall'])
    
    img = cv2.imread("AI/images/image_87.jpg")

    model.process_frame(img) # processes the image, finding (but not yet drawing) the results
    model.draw_results() # draws the results on the current processed frame based on the options set
    closest_ball = model.find_closest_ball("orange") # finds the closest ball of orange color
    model.highlight_ball(closest_ball) # highlights the closest orange ball in the current processed frame
    model.highlight_overlapping_boxes(exclude_classes=['wall']) # draws lines between overlapping boxes in the current processed frame, excluding walls

    objects = model.get_objects() # gets the objects found in the current processed frame
    print(objects) # prints the objects found in the current processed frame
    
    robot = objects['robot'][0]
    print(f"Robot is at center: {robot['center']} with bbox: {robot['bbox']} and confidence: {robot['confidence']:.2f}")

    path = model.plan_smooth_path(objects['egg'][0]['center'], obstacle_padding=10, max_curvature=0.05, num_waypoints=100)
    print(f"Planned path with {len(path)} waypoints.")
    model.draw_path(path) # draws the planned path on the current processed frame

    path = model.plan_smooth_path(objects['big_goal'][0]['center'], obstacle_padding=10, max_curvature=0.01, num_waypoints=100)
    print(f"Planned path to large goal with {len(path)} waypoints.")
    model.draw_path(path, color=(0, 255, 0), thickness=3) # draws the planned path to the large goal on the current processed frame

    # generate a path to each white ball
    
    #for ball in objects.get('white', []):
    #    path = model.plan_smooth_path(ball['center'], obstacle_padding=10, max_curvature=0.01, num_waypoints=100)
    #    print(f"Planned path to white ball at {ball['center']} with {len(path)} waypoints.")
    #    model.draw_path(path, color=(255, 0, 0), thickness=2) # draws the planned path to each white ball on the current processed frame

    cv2.imshow("Processed Frame", model.current_processed_drawn_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    img2 = cv2.imread("AI/images/image_26.jpg")
    model.process_frame(img2) # processes the second image
    model.draw_results() # draws the results on the second processed frame

    caught_ball = model.get_ball_overlapping_with_robot() # checks if a ball overlaps with the robot in the second processed frame
    if caught_ball:
        print(f"Caught ball: {caught_ball}")
        model.highlight_ball(caught_ball, False) # highlights the caught ball in the second processed frame
    else:
        print("No ball caught in the robot's claw.")
    
    cv2.imshow("Processed Frame 2", model.current_processed_drawn_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

