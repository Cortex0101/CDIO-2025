import math
import numpy as np
import cv2

class CourseObject:
    """
    Represents a detected object on the course, including its mask, bounding box,
    confidence score, label, and computed center point.
    """
    def __init__(self, label: str, mask: np.ndarray, bbox: tuple, confidence: float):
        self.label = label
        self.mask = mask  # List[ndarray] A list of segments in pixel coordinates.
        self.bbox = bbox  # (x1, y1, x2, y2) the box in xyxy format.
        self.confidence = confidence
        self.center = self._compute_center()

        # direction attribute only relevant for robot object
        self.direction = None

    def _compute_center(self):
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def __repr__(self):
        return (f"<CourseObject label={self.label!r} "
                f"conf={self.confidence:.2f} center={self.center}>")


class Course:
    """
    Container for all CourseObject instances detected in a single frame.
    Provides utilities to build from YOLO results and to query objects by label.
    """
    def __init__(self):
        self.objects = []  # List[CourseObject]
        self.width = 0
        self.height = 0

    @staticmethod 
    def _compute_direction(point1 : tuple, point2 : tuple) -> float:
        """
        Compute the direction vector from point1 to point2.

        Args:
            point1: (x1, y1) coordinates of the first point
            point2: (x2, y2) coordinates of the second point
        Returns:
            float: angle in degrees from point1 to point2
        """
        dx = point2[0] - point1[0]
        dy = point2[1] - point1[1]
        angle = (math.degrees(math.atan2(dy, dx)) + 360) % 360
        return angle

    @classmethod
    def from_yolo_results(cls, result):
        """
        Build a Course from a single ultralytics Result object.

        Args:
            result: ultralytics.engine.results.Results
        Returns:
            Course: populated with CourseObject instances
        """
        course = cls()
        masks = result.masks.xy 
        boxes = result.boxes.xyxy.cpu().numpy()  
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        names = result.names     

        for mask, bbox, conf, cls_id in zip(masks, boxes, confidences, class_ids):
            label = names.get(cls_id, str(cls_id))
            obj = CourseObject(label=label,
                               mask=mask,
                               bbox=tuple(bbox),
                               confidence=float(conf))
            course.objects.append(obj)

        # Compute direction for the robot if both yellow and green markers are present
        yellow = course.get_by_label('yellow')
        green = course.get_by_label('green')
        robot = course.get_by_label('robot')
        if yellow and green and robot:
            angle = cls._compute_direction(yellow[0].center,
                                           green[0].center)
            robot[0].direction = angle

        return course
    
    def is_complete(self):
        """
        Used to determine if the course is complete.
        A course is considered complete if it fulfills ALL the following conditions:
        - Contains exacly one robot object
        - Contains exactly one yellow and one green object
        - Contains at least two goals [can be either small or big, or both] (AI sometimes mistakes one for the other)
        - Contains exacly 1 wall object
        - Contains exacly 1 cross object

        This function is used when a stream of frames is processed, if suddenly the course is not complete anymore,
        we will have stored the last seen complete course, and we will assume that the objects missing have the 
        same properties as the last seen complete course.
        """
        robot = self.get_by_label('robot')
        yellow = self.get_by_label('yellow')
        green = self.get_by_label('green')
        small_goals = self.get_by_label('small_goal')
        big_goals = self.get_by_label('big_goal')
        walls = self.get_by_label('wall')
        crosses = self.get_by_label('cross')

        return (len(robot) == 1 and
                len(yellow) == 1 and
                len(green) == 1 and
                (len(small_goals) + len(big_goals)) == 2 and
                len(walls) == 1 and
                len(crosses) == 1)

    def get_by_label(self, label: str):
        """Return all CourseObjects matching a given label."""
        return [obj for obj in self.objects if obj.label == label]
    
    def get_robot(self):
        """ Returns robot object if it exists if course is complete, otherwise throws an error. """
        if not self.is_complete():
            raise ValueError("Course is not complete, cannot get robot object.")
        
        return self.get_by_label('robot')[0]
    
    def get_cross(self):
        """ Returns the cross object in the course. """
        if not self.is_complete():
            raise ValueError("Course is not complete, cannot get cross object.")
        
        return self.get_by_label('cross')[0]
    
    def get_floor(self):
        """ Returns the wall object, which is the floor in the course, only one if course is complete. """
        if not self.is_complete():
            raise ValueError("Course is not complete, cannot get floor object.")
       
        return self.get_by_label('wall')[0]

    def get_white_balls(self):
        """ Returns all white balls in the course. """
        return self.get_by_label('white')
    
    def get_orange_balls(self):
        """ Returns all orange balls in the course. """
        return self.get_by_label('orange')
    
    def get_eggs(self):
        """ Returns all eggs in the course. Might be more than one? """
        return self.get_by_label('egg')
    
    # TODO: functions for goals? Weird when they might be mixed up, or both small or both large

    def __iter__(self):
        return iter(self.objects)

    def __repr__(self):
        # print number of each object type
        counts = {}
        for obj in self.objects:
            counts[obj.label] = counts.get(obj.label, 0) + 1
        counts_str = ', '.join(f"{k}: {v}" for k, v in counts.items())
        return f"<Course objects=[{counts_str}]>"


class CourseVisualizer:
    TEXT_COLOR = (0, 255, 0)  # Green text color (distinguishable on most backgrounds)
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
    DEFAULT_COLOR = (255, 0, 0)  # Default to red if not found
    FONT = cv2.FONT_HERSHEY_TRIPLEX

    def __init__(self,
                 draw_centers: bool = True,
                 draw_center_points: bool = True,
                 draw_walls:   bool = False,
                 draw_direction_markers: bool = False,
                 draw_masks:   bool = False,
                 draw_boxes:   bool = True,
                 draw_confidence: bool = False,
                 draw_labels: bool = True,
                 mask_alpha:   float = 0.1):
        self.draw_centers = draw_centers
        self.draw_center_points = draw_center_points
        self.draw_walls   = draw_walls
        self.draw_direction_markers = draw_direction_markers
        self.draw_masks   = draw_masks
        self.draw_boxes   = draw_boxes
        self.draw_confidence = draw_confidence
        self.draw_labels  = draw_labels
        self.mask_alpha   = mask_alpha

    def draw(self, image: np.ndarray, course: Course) -> np.ndarray:
        """Return a new image with all the CourseObjects rendered on it."""
        canvas = image.copy()
        for obj in course:
            # 1) Optionally skip walls
            if obj.label == "wall" and not self.draw_walls:
                continue
            if obj.label in ["yellow", "green"] and not self.draw_direction_markers:
                continue

            # 2) Draw mask
            if self.draw_masks:
                pts = np.rint(obj.mask).astype(np.int32)
                pts = pts.reshape(-1, 1, 2)
                color = self.OBJECT_COLORS.get(obj.label, self.DEFAULT_COLOR)
                cv2.fillPoly(
                    canvas,
                    [pts],
                    color,
                    lineType=cv2.LINE_AA
                )
                cv2.polylines(
                    canvas,
                    [pts],
                    isClosed=True,
                    color=color,
                    thickness=2,
                    lineType=cv2.LINE_AA
                )
                
            # 3) Draw bbox
            if self.draw_boxes:
                x1,y1,x2,y2 = map(int, obj.bbox)
                cv2.rectangle(canvas, (x1,y1), (x2,y2), self.OBJECT_COLORS.get(obj.label, self.DEFAULT_COLOR) , 1)
                if self.draw_labels:
                    label_text = f"{obj.label} {obj.confidence:.2f}" if self.draw_confidence else obj.label # draw both?
                    cv2.putText(canvas, label_text, (x1, y1 - 10), self.FONT, 0.5, self.TEXT_COLOR, 1)
                elif self.draw_confidence:
                    cv2.putText(canvas, f"{obj.confidence:.2f}", (x1, y1 - 30), self.FONT, 0.5, self.TEXT_COLOR, 1)
                

            # 4) Draw center
            if self.draw_centers:
                cx, cy = map(int, obj.center)
                cv2.circle(canvas, (cx, cy), 3, self.OBJECT_COLORS.get(obj.label, self.DEFAULT_COLOR) , -1)
                if self.draw_center_points:
                    cv2.putText(canvas, f"({cx},{cy})", (cx + 5, cy + 10), self.FONT, 0.5, self.TEXT_COLOR, 1)

            # 5) Draw robot direction
            if obj.label == "robot" and obj.direction is not None:
                angle_rad = math.radians(obj.direction)
                length = 30
                end = (int(cx + length*math.cos(angle_rad)), int(cy + length*math.sin(angle_rad)))
                # cv2.arrowedLine(canvas, (cx, cy), end, (0,0,255), 2, tipLength=0.2)
                # draw arrow and text in top left corner
                cv2.arrowedLine(canvas, (cx, cy), end, self.OBJECT_COLORS.get(obj.label, self.DEFAULT_COLOR), 2, tipLength=0.2)
                cv2.putText(canvas, f"Direction: {obj.direction:.2f}°", (10, 30), self.FONT, 0.5, self.TEXT_COLOR, 1)

        return canvas

