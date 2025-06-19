import math
import logging
import numpy as np
import cv2

import config

'''
100 point pr bold der forlader banen gennem mål B
150 point pr. bold der forlader banen gennem mål A
200 points for at aflevere den orange bold først
3 point pr. resterende sekund når boldene er afleveret. 
-50 point hvis robotten berører banen/forhindringerne.
-100 hvis robotten flytter forhindringen/banen over 1 cm.
-300 for at flytte ægget mere end 1 cm.
'''

logger = logging.getLogger(__name__)

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
    def __init__(self, enable_errors: bool = False):
        self.objects = []  # List[CourseObject]
        self.width = 0
        self.height = 0
        self.enable_errors = enable_errors  # Enable error checking for course completeness

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
        if not result or not cls:
            logger.warning("No results to process, returning empty Course.")
            return cls()

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

        course.width = result.orig_shape[1]  # Width of the original image
        course.height = result.orig_shape[0]

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
        if not self.is_complete() and self.enable_errors:
            raise ValueError("Course is not complete, cannot get robot object.")
        
        if not self.get_by_label('robot'):
            return None
        return self.get_by_label('robot')[0]
    
    def get_cross(self):
        """ Returns the cross object in the course. """
        if not self.is_complete() and self.enable_errors:
            raise ValueError("Course is not complete, cannot get cross object.")
        
        if not self.get_by_label('cross'):
            return None
        return self.get_by_label('cross')[0]
    
    def get_floor(self):
        """ Returns the wall object, which is the floor in the course, only one if course is complete. """
        if not self.is_complete() and self.enable_errors:
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
    
    def get_goals(self):
        """
        Returns all goals in the course, both small and big.
        If there are no goals, returns an empty list.
        """
        return self.get_by_label('small_goal') + self.get_by_label('big_goal')

    # TODO: functions for goals? Weird when they might be mixed up, or both small or both large

    def get_optimal_ball_parking_spot(self, ball: CourseObject):
        """
        Find the optimal parking spot for a ball based on its position and the robot's position.
        Args:
            ball: CourseObject representing the ball to park
        Returns:
            tuple: (x, y) coordinates of the optimal parking spot
        """
        def _get_robot_size():
            """
            Get the size of the robot based on its bounding box.

            If a robot object is not found, it will return a default size.
            Returns:
                tuple: (width, height) of the robot
            """
            DEFAULT_ROBOT_SIZE = (90, 50)  # Default size if robot is not found (estimate from example image)
            robot = self.get_robot()
            if not robot:
                logger.warning("No robot found, using default robot size of %s.", DEFAULT_ROBOT_SIZE)
                return DEFAULT_ROBOT_SIZE
                
            return (robot.bbox[2] - robot.bbox[0], robot.bbox[3] - robot.bbox[1])
        
        robot_size = _get_robot_size()

        # determine if the balls is near a wall, corner or cross
        is_near_wall = self.is_ball_near_wall(ball)
        is_near_corner = self.is_ball_near_corner(ball)
        is_near_cross = self.is_ball_near_cross(ball)

        # if none are true, create a list of all points around the ball with a radius of max(robot_size)
        if not (is_near_wall or is_near_corner or is_near_cross):
            all_spots = []
            for angle in range(0, 360, 10):
                distance = max(robot_size) # using max in case the claw is closed or some weird situation, to ensure it can freely spin on the optimal spot
                # Calculate the new position based on the angle and distance
                circle_point = (ball.center[0] + distance * math.cos(math.radians(angle)),
                                ball.center[1] + distance * math.sin(math.radians(angle)))
                all_spots.append(circle_point)

            # sort the spots by distance to the robot
            all_spots.sort(key=lambda spot: np.linalg.norm(np.array(spot) - np.array(self.get_robot().center)))

            for spot in all_spots:
                # Get robot's bounding box at the circle point
                robot_bbox_at_circle_point = ( 
                    spot[0] - distance / 2, # using distance to avoid considering the robots rotation
                    spot[1] - distance / 2,
                    spot[0] + distance / 2,
                    spot[1] + distance / 2
                )
                # check if the robot's bounding box at the circle point lies within the course's floor bounding box
                if not self._bbox_lies_within_bbox(robot_bbox_at_circle_point, self.get_floor().bbox):
                    # if not, this point is not valid, continue to the next angle
                    continue
                # Check if the new spots bbox does not intersect with any other object
                overlaps_obstacle = False
                for obj in self.objects:
                    if obj.label != 'robot' and obj.label != 'wall' and obj is not ball:  # Exclude robot, wall and the ball itself
                        obj_bbox = obj.bbox
                        # Check if the robot's bounding box at the circle point intersects with the object's bounding box
                        if self._bbox_overlaps_bbox(robot_bbox_at_circle_point, obj_bbox):
                            overlaps_obstacle = True
                            break

                if not overlaps_obstacle:
                    # If the spot is valid, return it
                    return (int(spot[0]), int(spot[1]))

        # if robot is near a wall, but not near a corner or cross, find a a spot that is directly perpendicular to the wall the ball is near
        if is_near_wall and not (is_near_corner or is_near_cross):
            # ball: (np.float32(575.7888), np.float32(185.29161), np.float32(590.77075), np.float32(198.9276))
            # wall: (np.float32(69.766785), np.float32(39.335754), np.float32(589.8566), np.float32(417.1231))
            wall = self.get_floor()
            optimal_spot = None
            # Find which wall the ball is near
            wall_distances = {
                'left': abs(ball.bbox[0] - wall.bbox[0]),  # left wall
                'right': abs(ball.bbox[2] - wall.bbox[2]),  # right wall
                'top': abs(ball.bbox[1] - wall.bbox[1]),  # top wall
                'bottom': abs(ball.bbox[3] - wall.bbox[3])  # bottom wall
            }
            # Determine the closest wall
            closest_wall = min(wall_distances, key=wall_distances.get)
            distance = max(robot_size)  # Use the maximum dimension of the robot's bounding box
            if closest_wall == 'left':
                # Calculate the optimal spot to the left of the ball
                optimal_spot = (ball.center[0] + distance, ball.center[1])
            elif closest_wall == 'right':
                # Calculate the optimal spot to the right of the ball
                optimal_spot = (ball.center[0] - distance, ball.center[1])
            elif closest_wall == 'top':
                # Calculate the optimal spot above the ball
                optimal_spot = (ball.center[0], ball.center[1] + distance)
            elif closest_wall == 'bottom':
                # Calculate the optimal spot below the ball
                optimal_spot = (ball.center[0], ball.center[1] - distance)

            if optimal_spot:
                return (int(optimal_spot[0]), int(optimal_spot[1]))

        return (-1, -1) # If no valid spot is found, return (0, 0) or None

    def get_optimal_goal_parking_spot(self, goal_center: tuple): # should be a CourseObject but for debugging
        # for now just returns a spot that is 50 pixels to the side of the goal
        # in future should look for obstacles and use proper goal.
        floor = self.get_floor()
        wall_distances = {
                'left': abs(goal_center[0] - floor.bbox[0]),  # left wall
                'right': abs(goal_center[0] - floor.bbox[2]),  # right wall
                'top': abs(goal_center[1] - floor.bbox[1]),  # top wall
                'bottom': abs(goal_center[1] - floor.bbox[3])  # bottom wall
        }
        closest_wall = min(wall_distances, key=wall_distances.get)
        distance = 50  # Distance to park the goal away from the wall
        optimal_spot = None
        if closest_wall == 'left':
            # Calculate the optimal spot to the left of the goal
            optimal_spot = (goal_center[0] + distance, goal_center[1])
        elif closest_wall == 'right':
            # Calculate the optimal spot to the right of the goal
            optimal_spot = (goal_center[0] - distance, goal_center[1])
        elif closest_wall == 'top':
            # Calculate the optimal spot above the goal
            optimal_spot = (goal_center[0], goal_center[1] + distance)
        elif closest_wall == 'bottom':
            # Calculate the optimal spot below the goal
            optimal_spot = (goal_center[0], goal_center[1] - distance)

        return (int(optimal_spot[0]), int(optimal_spot[1])) if optimal_spot else None

    

    def prev_get_optimal_ball_parking_spot(self, ball: CourseObject):
        """
        Find the optimal parking spot for a ball based on its position and the robot's position.
        
        This functions looks in 360 degrees around the balls position (at a distance of the size of the robots bounding box), and checks if there is a spot, the size
        of the robots bounding box, that is not occupied by any other object (minimizing the possibility of collisions).

        If several spots are found, it will return the one closest to the robot.
        If none is found, it will return None and we can look for another ball.

        Args:
            ball: CourseObject representing the ball to park
        Returns:
            tuple: (x, y) coordinates of the optimal parking spot
        """
        def _get_robot_size():
            """
            Get the size of the robot based on its bounding box.

            If a robot object is not found, it will return a default size.
            Returns:
                tuple: (width, height) of the robot
            """
            DEFAULT_ROBOT_SIZE = (90, 50)  # Default size if robot is not found (estimate from example image)
            robot = self.get_robot()
            if not robot:
                logger.warning("No robot found, using default robot size of %s.", DEFAULT_ROBOT_SIZE)
                return DEFAULT_ROBOT_SIZE
                
            return (robot.bbox[2] - robot.bbox[0], robot.bbox[3] - robot.bbox[1])

        robot_size = _get_robot_size()
        ball_center = ball.center

        optimal_spot = None
        min_distance = float('inf')
        for angle in range(0, 360, 10):  # Check every 10 degrees
            # Calculate the new position based on the angle and distance
            distance = min(robot_size)  # Use the smaller dimension of the robot's bounding box
            x = ball_center[0] + distance * math.cos(math.radians(angle))
            y = ball_center[1] + distance * math.sin(math.radians(angle))
            new_spot = (x, y)

            # Check if the new spot is within the course boundaries
            if (0 <= x <= self.width and 0 <= y <= self.height):
                # Check if the new spot is not occupied by any other object
                is_occupied = False
                for obj in self.objects:
                    if obj.label != 'robot' and obj.label != 'wall':
                        obj_bbox = obj.bbox
                        if (obj_bbox[0] <= x <= obj_bbox[2] and
                            obj_bbox[1] <= y <= obj_bbox[3]):
                            is_occupied = True
                            break

                if not is_occupied:
                    # Calculate distance to the robot
                    distance_to_robot = np.linalg.norm(np.array(new_spot) - np.array(robot.center))
                    if distance_to_robot < min_distance:
                        min_distance = distance_to_robot
                        optimal_spot = new_spot

        return (float(optimal_spot[0]), float(optimal_spot[1])) if optimal_spot else None

    def get_nearest_ball(self, point: tuple, excluded=[], color: str = 'white'):
        """
        Find the nearest white ball to a given point.

        Args:
            point: (x, y) coordinates of the point to search from
        Returns:
            CourseObject: the nearest white ball object, or None if none found
        """
        balls = []

        if color == 'orange':
            balls = self.get_orange_balls()
        elif color == 'white':
            balls = self.get_white_balls()
        elif color == 'either':
            balls = self.get_white_balls() + self.get_orange_balls()
        else:
            raise ValueError(f"Unknown color: {color}. Use 'white', 'orange', or 'either'.")

        if not balls:
            return None
        
        # Filter out excluded balls
        balls = [ball for ball in balls if ball not in excluded]

        nearest_ball = min(balls, key=lambda obj: np.linalg.norm(np.array(obj.center) - np.array(point)))
        return nearest_ball

    def get_nearest_goal(self, point: tuple):
        """
        Find the nearest goal to a given point.

        Args:
            point: (x, y) coordinates of the point to search from
        Returns:
            CourseObject: the nearest goal object, or None if none found
        """
        goals = self.get_by_label('small_goal') + self.get_by_label('big_goal')
        if not goals:
            return None

        nearest_goal = min(goals, key=lambda obj: np.linalg.norm(np.array(obj.center) - np.array(point)))
        return nearest_goal

    def get_nearest_object(self, point: tuple):
        """
        Find the nearest object to a given point, that is not robot, wall, or cross.

        Args:
            point: (x, y) coordinates of the point to search from
        Returns:
            CourseObject: the nearest object, or None if no objects found
        """
        if not self.objects:
            return None
        
        # Filter out robot, wall, and cross objects
        filtered_objects = [obj for obj in self.objects if obj.label not in ['robot', 'wall', 'cross']]

        if not filtered_objects:
            return None
        
        nearest_object = min(filtered_objects, key=lambda obj: np.linalg.norm(np.array(obj.center) - np.array(point)))
        return nearest_object

    def is_ball_near_wall(self, ball: CourseObject, threshold: int = config.LARGE_OBJECT_RADIUS):
        """
        Check if a ball is near the edge of the floor. 
        That is, a ball is considered near the wall if it is bbox lies within a certain threshold distance
        from the wall's bounding box.

        Args:
            ball: CourseObject representing the ball to check
        Returns:
            bool: True if the ball is near the wall, False otherwise
        """
        wall = self.get_floor()
        if not wall:
            return False

        return self._bbox_within_threshold_bbox(ball.bbox, wall.bbox, threshold)

    def is_ball_near_corner(self, ball: CourseObject, threshold: int = config.LARGE_OBJECT_RADIUS):
        """
        Check if a ball is near the corner of the floor.

        Args:
            ball: CourseObject representing the ball to check
        Returns:
            bool: True if the ball is near the corner, False otherwise
        """
        wall = self.get_floor()
        if not wall:
            return False

        # Define the corners of the wall
        corners = [
            (wall.bbox[0], wall.bbox[1]),  # Top-left
            (wall.bbox[2], wall.bbox[1]),  # Top-right
            (wall.bbox[0], wall.bbox[3]),  # Bottom-left
            (wall.bbox[2], wall.bbox[3])   # Bottom-right
        ]

        # Check if the ball's bounding box is within threshold distance from any two corners
        for corner in corners:
            if self._bbox_within_threshold_point(ball.bbox, corner, threshold):
                return True
            
        return False

    def is_ball_near_cross(self, ball: CourseObject, threshold: int = config.SMALL_OBJECT_RADIUS):
        """
        Check if a ball is near the cross object.

        Args:
            ball: CourseObject representing the ball to check
            threshold: distance threshold to consider as "near"
        Returns:
            bool: True if the ball is near the cross, False otherwise
        """
        cross = self.get_cross()
        if not cross:
            return False

        return self._bbox_within_threshold_bbox(ball.bbox, cross.bbox, threshold)

    def _bbox_lies_within_bbox(self, inner_bbox: tuple, outer_bbox: tuple) -> bool:
        """
        Check if the CourseObject's bounding box lies within a given bounding box.

        Args:
            obj: CourseObject to check
            bbox: (x1, y1, x2, y2) bounding box coordinates to check against
        Returns:
            bool: True if the object's bbox lies within the given bbox, False otherwise
        """
        return (outer_bbox[0] <= inner_bbox[1] <= outer_bbox[2] and
                outer_bbox[1] <= inner_bbox[1] <= outer_bbox[3] and
                outer_bbox[0] <= inner_bbox[1] <= outer_bbox[2] and
                outer_bbox[1] <= inner_bbox[1] <= outer_bbox[3])

    def _bbox_overlaps_bbox(self, bbox1: tuple, bbox2: tuple) -> bool:
        """
        Check if two bounding boxes overlap.

        Args:
            bbox1: (x1, y1, x2, y2) bounding box coordinates of the first object
            bbox2: (x1, y1, x2, y2) bounding box coordinates of the second object
        Returns:
            bool: True if the bounding boxes overlap, False otherwise
        """
        return not (bbox1[0] > bbox2[2] or bbox1[2] < bbox2[0] or
                    bbox1[1] > bbox2[3] or bbox1[3] < bbox2[1])

    def _bbox_within_threshold_bbox(self, bbox1: tuple, bbox2: tuple, threshold: int = 0) -> bool:
        """
        Check if the bounding box bbox1 is within a certain threshold distance from bbox2.

        Args:
            bbox1: (x1, y1, x2, y2) bounding box coordinates of the first object
            bbox2: (x1, y1, x2, y2) bounding box coordinates of the second object
            threshold: distance threshold to consider as "near"
        Returns:
            bool: True if some part of bbox1 is within threshold distance from bbox2, False otherwise
        """
        return (abs(bbox1[0] - bbox2[0]) <= threshold or
                abs(bbox1[1] - bbox2[1]) <= threshold or
                abs(bbox1[2] - bbox2[2]) <= threshold or
                abs(bbox1[3] - bbox2[3]) <= threshold)
    
    def _bbox_within_threshold_point(self, bbox: tuple, point: tuple, threshold: int = 0) -> bool:
        """
        Check if the bounding box is within a certain threshold distance from a point.
        Args:
            bbox: (x1, y1, x2, y2) bounding box coordinates of the object
            point: (x, y) coordinates of the point to check against
            threshold: distance threshold to consider as "near"
        Returns:
            bool: True if the bounding box is within threshold distance from the point, False otherwise
        """
        x1, y1, x2, y2 = bbox
        x, y = point
        return (x1 - threshold <= x <= x2 + threshold and
                y1 - threshold <= y <= y2 + threshold)

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
    BALL_HIGHLIGHT_COLOR = (0, 255, 255)  # Cyan for ball highlight
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
    
    def highlight_ball(self, image: np.ndarray, ball: CourseObject, color: tuple = None) -> np.ndarray:
        """
        Highlight a specific ball in the image by drawing a bounding box and label.

        Args:
            image: The original image to draw on.
            ball: The CourseObject representing the ball to highlight.
        Returns:
            np.ndarray: The modified image with the highlighted ball.
        """
        if color is None:
            color = self.BALL_HIGHLIGHT_COLOR
        canvas = image.copy()
        x1, y1, x2, y2 = map(int, ball.bbox)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        label_text = f"{ball.label} {ball.confidence:.2f}" if self.draw_confidence else ball.label
        cv2.putText(canvas, label_text, (x1, y1 - 10), self.FONT, 0.5, self.TEXT_COLOR, 1)
        return canvas
    
    def highlight_point(self, image: np.ndarray, point: tuple, color: tuple = (0, 255, 0), radius: int = 5) -> np.ndarray:
        """
        Highlight a specific point in the image.

        Args:
            image: The original image to draw on.
            point: (x, y) coordinates of the point to highlight.
            color: Color of the highlight circle.
            radius: Radius of the highlight circle.
        Returns:
            np.ndarray: The modified image with the highlighted point.
        """
        canvas = image.copy()
        cv2.circle(canvas, (int(point[0]), int(point[1])), radius, color, -1)
        return canvas
