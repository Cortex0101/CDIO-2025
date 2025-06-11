from ultralytics import YOLO
import cv2
import numpy as np
from scipy.interpolate import splprep, splev
import heapq

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

    def stream_to_model_results_to_course_objects(self, model_results):
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
            self.add_object(obj)
        
        return objects

    def add_object(self, obj: CourseObject):
        self.objects.append(obj)
        if obj.name in self.object_count:
            self.object_count[obj.name] += 1
        else:
            self.object_count[obj.name] = 1

    def get_objects_by_name(self, name):
        objects = []
        for obj in self.objects:
            if obj.name == name:
                objects.append(obj)

        return objects

    def __repr__(self):
        return f"Course with {len(self.objects)} objects: {self.object_count}"

class AIModel:
    def __init__(self):
        self.model = YOLO("ball_detect/v8/weights/best.pt")
        
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
            "egg": (238, 130, 238),      # Violet - easily distinguishable, rarely confused
            "robot": (0, 200, 70),       # Medium Green - stands out from background and other objects
            "white_ball": (220, 220, 220), # Light Gray - more visible than pure white
            "orange_ball": (255, 140, 0),  # Rich Orange - high contrast, classic orange
            "small_goal": (70, 130, 180),  # Steel Blue - visually distinct from red/yellow
            "big_goal": (255, 215, 0),   # Gold - deep yellow, more visible than light yellow
            "wall": (70, 70, 70),          # Dark Gray - strong contrast for obstacles
            "cross": (75, 0, 130),         # Indigo - deep blue/purple, distinct from wall/robot
            "yellow": (255, 255, 0),       # Bright Yellow - classic, high contrast
            "green": (50, 205, 50),        # Lime Green - distinct from robot's green
            "empty": (200, 200, 200)                # Light Gray - neutral color for empty cells
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

        self.current_course = Course()  # Reset current course for new frame
        self.current_course.stream_to_model_results_to_course_objects(self.current_results)
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

    def determine_robot_direction(self):
        """
        Determines the direction of the robot based on the object with 'yellow' label being the back,
        and the object with 'green' label being the front.

        Draw a line from the center of the yellow object to the center of the green object.

        The direction vector between those two points will be the direction of the robot.
        
        Returns:
            int: A number from 0 being to the right, 90 being up, 180 being to the left, and 270 being down.
        """
        yellow_objects = self.current_course.get_objects_by_name("yellow")
        green_objects = self.current_course.get_objects_by_name("green")

        if not yellow_objects or not green_objects:
            print("Cannot determine robot direction, missing green or yellow object.")
            return None
        
        yellow_center = yellow_objects[0].center
        green_center = green_objects[0].center

        # Calculate the direction vector from green to yellow
        direction_vector =  np.array(green_center) - np.array(yellow_center)
        angle = np.arctan2(direction_vector[1], direction_vector[0])  # Angle in radians
        angle = np.degrees(angle)  # Convert to degrees
        angle = (angle + 360) % 360  # Normalize to [0, 360)

        return int(angle)
    
    def draw_robot_direction(self):
        """
        Draws the direction of the robot on the current processed frame.
        The direction is determined by the yellow and green objects.
        """
        angle = self.determine_robot_direction()
        if angle is None:
            return
        
        # Draw an arrow from the center of the green object to the center of the yellow object
        green_objects = self.current_course.get_objects_by_name("green")
        yellow_objects = self.current_course.get_objects_by_name("yellow")
        if not green_objects or not yellow_objects:
            print("Cannot draw robot direction, missing green or yellow object.")
            return
        
        green_center = green_objects[0].center
        yellow_center = yellow_objects[0].center
        cv2.arrowedLine(self.current_processed_drawn_frame,
                        (int(yellow_center[0]), int(yellow_center[1])),
                        (int(green_center[0]), int(green_center[1])),
                        (255, 0, 0), 2, tipLength=0.1)
        
    def find_closest_ball(self, ball_type: str = "white") -> dict | None:
        """
        Finds the closest ball of the specified color (or if "either", any ball) in the current results.

        Args:
            ball_color: Color of the ball to find (e.g., "white", "orange" or "either").

        Returns:
            A dict with 'class', 'confidence', 'bbox', 'centroid' if found, else None.
        """
        closest_ball = None
        min_distance = float('inf')

        for obj in self.current_course.objects:
            if obj.name not in [ball_type, "either"] and ball_type != "either":
                continue
            
            # Calculate distance from the robot to the ball
            robot = self.current_course.get_objects_by_name("robot")[0]
            if not robot:
                continue
            
            robot_center = robot.center
            ball_center = obj.center
            distance = np.linalg.norm(np.array(robot_center) - np.array(ball_center))

            if distance < min_distance:
                min_distance = distance
                closest_ball = obj

        return closest_ball if closest_ball else None
    
    def highlight_ball(self, ball): 
        """
        Highlights the closest ball in the current processed frame by drawing a rectangle and label on that frame.
        This method modifies the `self.current_processed_drawn_frame` attribute.

        Args:
            ball: Dict with 'class', 'confidence', 'bbox', 'centroid'.
        """
        # bright green
        highlight_color = (0, 165, 255)
        box = ball.bbox
        cv2.rectangle(self.current_processed_drawn_frame,
                        (box[0], box[1]), (box[2], box[3]), highlight_color, 2)
        
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
    
    def plan_smooth_path_from_robot(self, target_coord, obstacle_padding=5, max_curvature=0.05, num_waypoints=100):
        """
        Generate a smooth, obstacle-avoiding path from robot to target using bounding boxes.

        Args:
            target_coord: (x,y) goal center.
            obstacle_padding: extra padding around obstacles.
            max_curvature: smoothing factor (lower -> tighter turns).
            num_waypoints: number of sampled points.

        Returns:
            smooth_path: ndarray of shape (N,2) of x,y waypoints.
        """
        robot = self.current_course.get_objects_by_name("robot")[0]
        return self.plan_smooth_path(robot.center, target_coord, obstacle_padding, max_curvature, num_waypoints)

    def plan_smooth_path(self, start_coord, target_coord, obstacle_padding=5, max_curvature=0.05, num_waypoints=100):
        """
        Generate a smooth, obstacle-avoiding path from robot to target using bounding boxes.

        Args:
            target_coord: (x,y) goal center.
            obstacle_padding: extra padding around obstacles.
            max_curvature: smoothing factor (lower -> tighter turns).
            num_waypoints: number of sampled points.

        Returns:
            smooth_path: ndarray of shape (N,2) of x,y waypoints.
        """
        # get all objects in the current course and remove the robot and the object with the same center as the target
        filtered_objects = []
        for obj in self.current_course.objects:
            if obj.name == "robot" or obj.center == target_coord:
                continue

            filtered_objects.append(obj)

        # 1. Sample straight-line path
        robot = self.current_course.get_objects_by_name("robot")[0]
        t = np.linspace(0, 1, num_waypoints)
        line = np.outer(1 - t, start_coord) + np.outer(t, target_coord)
        robot_radius = (robot.bbox[2] - robot.bbox[0]) / 2.0  # use half of width as radius

        # 2. Build obstacle circles (center, radius)
        circles = []
        for obj in filtered_objects:
            if obj.name in self.excluded_classes:
                continue
            
            cx = (obj.bbox[0] + obj.bbox[2]) / 2.0
            cy = (obj.bbox[1] + obj.bbox[3]) / 2.0
            # approximate radius as half of diagonal
            r = np.hypot(obj.bbox[2] - obj.bbox[0], obj.bbox[3] - obj.bbox[1]) / 2.0
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

        # Remove duplicate points
        _, unique_indices = np.unique(path, axis=0, return_index=True)
        unique_path = path[np.sort(unique_indices)]

        # Check if enough unique points for splprep (k=3 needs at least 4)
        if unique_path.shape[0] < 4:
            # fallback: just return the straight line
            return path

        # 4. Interpolate smooth spline
        tck, u = splprep([unique_path[:, 0], unique_path[:, 1]], s=max_curvature * num_waypoints)
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

    def get_balls_overlapping_with_robot(self): 
        """
        Finds the ball that overlaps with the robot in the current processed frame.

        Returns:
            dict: Ball information if found, else None.
        """
        robot = self.current_course.get_objects_by_name("robot")[0]
        if not robot:
            return None

        overlapping_balls = []

        for obj in self.current_course.objects:
            if obj.name not in ['white', 'orange']:
                continue
            
            ball = obj

            if self.do_boxes_overlap(robot.bbox, ball.bbox):
                overlapping_balls.append(ball)

        return overlapping_balls

    def determine_most_optimal_ball(self, balls): #TODO: Finish
        """
        Determines the most optimal ball to target based on some criteria.
        For now, this method simply returns the first ball in the list.

        Args:
            balls: List of balls to choose from.

        Returns:
            The most optimal ball.
        """
        if not balls:
            return None
        
        # generate path to each ball and then a path from that ball to the closest goal
        for ball in balls:
            path_to_ball = self.plan_smooth_path(ball.center, obstacle_padding=10, max_curvature=0.01, num_waypoints=100)
            path_from_ball_to_goal = self.plan_smooth_path(self.current_course.get_objects_by_name('large_goal')[0].center, obstacle_padding=10, max_curvature=0.01, num_waypoints=100)

    def create_grid_from_image(self, cell_size):
        """
        Creates a grid cell based representation of the current processed frame.

        Each object in the course is represented by filling the grid cells with a label or 'empty' if no object is present.
        
        Args:
            cell_size (int): Size of each grid cell in pixels.
        
        Returns:
            grid_image (ndarray): 2x2 grid where each cell is filled with a string representing the object type.
        """
        height, width = self.current_processed_drawn_frame.shape[:2]
        grid_height = height // cell_size
        grid_width = width // cell_size

        grid_image = np.full((grid_height, grid_width), 'wall', dtype=object) # The cells called "walls" are actually empty, and empty are actually walls

        for obj in self.current_course.objects:
            # Calculate the grid cell coordinates
            x1, y1, x2, y2 = obj.bbox
            cell_x1 = x1 // cell_size
            cell_y1 = y1 // cell_size
            cell_x2 = x2 // cell_size
            cell_y2 = y2 // cell_size
            
            # Fill the grid cells with the object name
            for i in range(cell_x1, min(cell_x2 + 1, grid_width)):
                for j in range(cell_y1, min(cell_y2 + 1, grid_height)):
                    if obj.name == 'wall':
                        grid_image[j, i] = 'empty'  # The cells called "walls" are actually empty, and empty are actually walls
                    else:
                        grid_image[j, i] = obj.name
        
        return grid_image
    
    def convert_grid_to_grid_image(self, grid_image):
        """
        Displays the grid image in a window.
        
        Args:
            grid_image (ndarray): The grid image to display.
        """
        # Convert grid to a color image for visualization
        color_grid = np.zeros((grid_image.shape[0], grid_image.shape[1], 3), dtype=np.uint8)
        for i in range(grid_image.shape[0]):
            for j in range(grid_image.shape[1]):
                color = self.COLORS.get(grid_image[i, j], (255, 255, 255))
                color_grid[i, j] = color
        # Resize for better visibility
        return color_grid
  
    def pixel_to_grid(self, pixel_coord, cell_size):
        """Convert pixel coordinates to grid coordinates."""
        x, y = pixel_coord
        return int(x // cell_size), int(y // cell_size)
    
    def grid_to_pixel(self, grid_coord, cell_size):
        """Convert grid coordinates to pixel coordinates (center of cell)."""
        gx, gy = grid_coord
        return int(gx * cell_size + cell_size // 2), int(gy * cell_size + cell_size // 2)
    
    def get_neighbors(self, pos, grid):
        """Get valid neighboring cells for pathfinding."""
        x, y = pos
        neighbors = []
        grid_height, grid_width = grid.shape
        
        # 8-directional movement (including diagonals)
        directions = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if (0 <= nx < grid_width and 0 <= ny < grid_height and 
                grid[ny, nx] == 0):  # Passable cell
                # Calculate cost (diagonal moves cost more)
                cost = 1.414 if abs(dx) + abs(dy) == 2 else 1.0
                neighbors.append(((nx, ny), cost))
        
        return neighbors
    
    def dijkstra_pathfind(self, start_pos, end_pos, grid):
        """
        Find shortest path using Dijkstra's algorithm.
        """
        # Priority queue: (cost, current_pos)
        pq = [(0, start_pos)]
        distances = {start_pos: 0}
        previous = {}
        visited = set()
        
        while pq:
            current_cost, current_pos = heapq.heappop(pq)
            
            if current_pos in visited:
                continue
                
            visited.add(current_pos)
            
            # Found the target
            if current_pos == end_pos:
                # Reconstruct path
                path = []
                pos = end_pos
                while pos is not None:
                    path.append(pos)
                    pos = previous.get(pos)
                path.reverse()
                return path, distances[end_pos]
            
            # Check neighbors
            for neighbor_pos, move_cost in self.get_neighbors(current_pos, grid):
                if neighbor_pos in visited:
                    continue
                
                new_cost = current_cost + move_cost
                
                if neighbor_pos not in distances or new_cost < distances[neighbor_pos]:
                    distances[neighbor_pos] = new_cost
                    previous[neighbor_pos] = current_pos
                    heapq.heappush(pq, (new_cost, neighbor_pos))
        
        # No path found
        return None, float('inf')
    
    def find_paths_to_all_balls(self, cell_size=20):
        """
        Find shortest paths to all balls using Dijkstra's algorithm.
        """
        # Get robot position
        robot_objects = self.current_course.get_objects_by_name("robot")
        if not robot_objects:
            return {}
        
        robot = robot_objects[0]
        
        # Create navigation grid
        grid, grid_width, grid_height, cell_size = self.create_navigation_grid(cell_size)
        
        # Convert robot position to grid coordinates
        robot_grid_pos = self.pixel_to_grid(robot.center, cell_size)
        
        paths_to_balls = {}
        
        # Find all balls
        ball_types = ['white_ball', 'orange_ball', 'white', 'orange']
        for ball_type in ball_types:
            balls = self.current_course.get_objects_by_name(ball_type)
            for ball in balls:
                # Convert ball position to grid coordinates
                ball_grid_pos = self.pixel_to_grid(ball.center, cell_size)
                
                # Check if positions are valid
                if (0 <= robot_grid_pos[0] < grid_width and 0 <= robot_grid_pos[1] < grid_height and
                    0 <= ball_grid_pos[0] < grid_width and 0 <= ball_grid_pos[1] < grid_height):
                    
                    # Find path using Dijkstra
                    path_grid, cost = self.dijkstra_pathfind(robot_grid_pos, ball_grid_pos, grid)
                    
                    if path_grid:
                        # Convert path back to pixel coordinates
                        path_pixels = [self.grid_to_pixel(pos, cell_size) for pos in path_grid]
                        paths_to_balls[ball] = (path_pixels, cost)
        
        return paths_to_balls
    
    def create_navigation_grid(self, cell_size=20):
        """
        Creates a navigation grid for pathfinding where each cell represents
        whether it's passable (0) or blocked (1).
        
        IMPORTANT: Objects marked as 'wall' are actually the DRIVEABLE TRACK!
        """
        height, width = self.current_processed_drawn_frame.shape[:2]
        grid_height = height // cell_size
        grid_width = width // cell_size
        
        # Initialize grid as blocked (everything off-track is not driveable)
        grid = np.ones((grid_height, grid_width), dtype=int)
        
        # Mark 'wall' objects as PASSABLE (they are the track!)
        track_objects = self.current_course.get_objects_by_name('wall')
        passable_cells = 0
        
        for track_obj in track_objects:
            x1, y1, x2, y2 = track_obj.bbox
            
            # Convert to grid coordinates
            cell_x1 = max(0, x1 // cell_size)
            cell_y1 = max(0, y1 // cell_size)
            cell_x2 = min(grid_width - 1, x2 // cell_size)
            cell_y2 = min(grid_height - 1, y2 // cell_size)
            
            # Mark track cells as passable
            for i in range(int(cell_x1), int(cell_x2) + 1):
                for j in range(int(cell_y1), int(cell_y2) + 1):
                    if 0 <= i < grid_width and 0 <= j < grid_height:
                        if grid[j, i] == 1:  # Only count if we're changing from blocked to passable
                            passable_cells += 1
                        grid[j, i] = 0  # Passable (track surface)
        
        # Now mark actual obstacles as blocked
        actual_obstacles = ['egg']  # Add other obstacle types that should block movement
        blocked_cells = 0
        
        for obj in self.current_course.objects:
            if obj.name in actual_obstacles:
                x1, y1, x2, y2 = obj.bbox
                
                # Add padding around obstacles for robot clearance
                padding = 2  # Adjust based on robot size
                cell_x1 = max(0, (x1 // cell_size) - padding)
                cell_y1 = max(0, (y1 // cell_size) - padding)
                cell_x2 = min(grid_width - 1, (x2 // cell_size) + padding)
                cell_y2 = min(grid_height - 1, (y2 // cell_size) + padding)
                
                # Mark obstacle cells as blocked
                for i in range(int(cell_x1), int(cell_x2) + 1):
                    for j in range(int(cell_y1), int(cell_y2) + 1):
                        if 0 <= i < grid_width and 0 <= j < grid_height:
                            if grid[j, i] == 0:  # Only block if it was previously passable
                                blocked_cells += 1
                            grid[j, i] = 1  # Blocked
        
        # Debug information
        total_cells = grid_width * grid_height
        final_passable = np.sum(grid == 0)
        final_blocked = np.sum(grid == 1)
        
        print(f"Navigation grid created: {grid_width}x{grid_height} = {total_cells} total cells")
        print(f"Track objects found: {len(track_objects)} (marked as passable)")
        print(f"Passable cells: {final_passable} ({(final_passable/total_cells)*100:.1f}%)")
        print(f"Blocked cells: {final_blocked} ({(final_blocked/total_cells)*100:.1f}%)")
        print(f"Obstacle types: {actual_obstacles}")
        
        return grid, grid_width, grid_height, cell_size

    def draw_dijkstra_paths(self, cell_size=20):
        """
        Draw Dijkstra paths to all balls with improved visualization and distance display.
        """
        # Get robot position first for distance calculations
        robot_objects = self.current_course.get_objects_by_name("robot")
        if not robot_objects:
            print("No robot found - cannot calculate distances")
            return
        
        robot = robot_objects[0]
        
        # Find all balls and calculate straight-line distances
        ball_types = ['white_ball', 'orange_ball', 'white', 'orange']
        all_balls = []
        for ball_type in ball_types:
            balls = self.current_course.get_objects_by_name(ball_type)
            all_balls.extend(balls)
        
        if not all_balls:
            print("No balls found - cannot show distances")
            return
        
        # Calculate and display straight-line distances to each ball
        print(f"\nDistances from robot at {robot.center}:")
        ball_distances = []
        
        for i, ball in enumerate(all_balls):
            # Calculate straight-line distance
            distance = np.linalg.norm(np.array(robot.center) - np.array(ball.center))
            ball_distances.append((ball, distance))
            print(f"Ball {i+1} ({ball.name}) at {ball.center}: {distance:.1f} pixels")
            
            # Draw distance label on each ball
            ball_center = (int(ball.center[0]), int(ball.center[1]))
            distance_text = f"{distance:.0f}px"
            
            # Draw background for text
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            font_thickness = 2
            text_size = cv2.getTextSize(distance_text, font, font_scale, font_thickness)[0]
            
            # Position text above the ball
            text_x = ball_center[0] - text_size[0] // 2
            text_y = ball_center[1] - 15
            
            # Draw background rectangle
            cv2.rectangle(self.current_processed_drawn_frame,
                        (text_x - 3, text_y - text_size[1] - 3),
                        (text_x + text_size[0] + 3, text_y + 3),
                        (0, 0, 0), -1)  # Black background
            
            # Draw distance text
            cv2.putText(self.current_processed_drawn_frame, distance_text,
                    (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness)
        
        # Sort by distance and show closest ball
        ball_distances.sort(key=lambda x: x[1])
        closest_ball, closest_distance = ball_distances[0]
        print(f"\nClosest ball: {closest_ball.name} at distance {closest_distance:.1f} pixels")
        
        # Now try to find pathfinding paths
        paths_to_balls = self.find_paths_to_all_balls(cell_size)
        
        # Color scheme for different paths
        colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
            (128, 0, 128),  # Purple
            (255, 165, 0),  # Orange
        ]
        
        print(f"\nFound {len(paths_to_balls)} pathfinding routes")
        
        if len(paths_to_balls) == 0:
            print("No pathfinding routes found - this might be due to grid obstacles")
            # Show navigation grid for debugging
            self.draw_navigation_grid(cell_size=cell_size, alpha=0.3)
        else:
            # Sort paths by cost to show shortest path first
            sorted_paths = sorted(paths_to_balls.items(), key=lambda x: x[1][1])
            
            for i, (ball, (path_pixels, cost)) in enumerate(sorted_paths):
                print(f"Pathfinding route {i+1} to {ball.name} ball: cost {cost:.1f}")
                
                if len(path_pixels) < 2:
                    continue
                
                color = colors[i % len(colors)]
                thickness = max(2, 5 - i)
                
                # Draw path
                for j in range(len(path_pixels) - 1):
                    cv2.line(self.current_processed_drawn_frame, 
                            path_pixels[j], path_pixels[j + 1], color, thickness)
                
                # Draw arrow at the end
                if len(path_pixels) >= 2:
                    end_point = path_pixels[-1]
                    second_last = path_pixels[-2]
                    cv2.arrowedLine(self.current_processed_drawn_frame,
                                second_last, end_point, color, thickness + 1, tipLength=0.3)

    def draw_navigation_grid(self, cell_size=20, alpha=0.3):
        """
        Draw the navigation grid overlay for debugging with correct visualization.
        Green = driveable track, Red = obstacles/off-track areas
        """
        grid, grid_width, grid_height, _ = self.create_navigation_grid(cell_size)
        
        # Create overlay
        overlay = self.current_processed_drawn_frame.copy()
        
        for i in range(grid_width):
            for j in range(grid_height):
                x1 = i * cell_size
                y1 = j * cell_size
                x2 = x1 + cell_size
                y2 = y1 + cell_size
                
                if grid[j, i] == 1:  # Blocked cell (off-track or obstacle)
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), -1)  # Red fill
                else:  # Passable cell (driveable track)
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 1)  # Green outline
        
        # Blend with original image
        cv2.addWeighted(overlay, alpha, self.current_processed_drawn_frame, 1 - alpha, 0, 
                    self.current_processed_drawn_frame)

def demo():
    model = AIModel()

    model.set_options(show_boxes=True, show_masks=False, show_confidence=False, show_labels=False, show_center=False)
    model.set_excluded_classes(['wall'])

    img = cv2.imread("AI/images/image_65.jpg")
    model.process_frame(img) # processes the third image
    model.draw_results() # draws the results on the third processed frame

    model.highlight_ball(model.current_course.get_objects_by_name('white')[3]) # highlights the first white ball in the third processed frame
    path = model.plan_smooth_path_from_robot(model.current_course.get_objects_by_name('white')[3].center, obstacle_padding=10, max_curvature=0.05, num_waypoints=100)
    model.draw_path(path, color=(255, 0, 0), thickness=2) # draws the planned path to the first white ball on the current processed frame

    path = model.plan_smooth_path(model.current_course.get_objects_by_name('white')[0].center, model.current_course.get_objects_by_name('white')[1].center, obstacle_padding=10, max_curvature=0.01, num_waypoints=100)
    model.draw_path(path, color=(0, 255, 20), thickness=3) # draws the planned path from the first white ball to the large goal on the current processed frame

    cv2.imshow("Processed Frame 3", model.current_processed_drawn_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def demo1():
    model = AIModel()

    model.set_options(show_boxes=True, show_masks=False, show_confidence=False, show_labels=True, show_center=False)
    model.set_excluded_classes(['wall'])
    
    img = cv2.imread("AI/images/image_87.jpg")

    model.process_frame(img) # processes the image, finding (but not yet drawing) the results
    model.draw_results() # draws the results on the current processed frame based on the options set
    closest_ball = model.find_closest_ball("white") # finds the closest ball of orange color
    model.highlight_ball(closest_ball) # highlights the closest orange ball in the current processed frame
    model.highlight_overlapping_boxes(exclude_classes=['wall']) # draws lines between overlapping boxes in the current processed frame, excluding walls

    objects = model.get_objects() # gets the objects found in the current processed frame
    print(objects) # prints the objects found in the current processed frame
    
    robot = objects['robot'][0]
    print(f"Robot is at center: {robot['center']} with bbox: {robot['bbox']} and confidence: {robot['confidence']:.2f}")

    path = model.plan_smooth_path_from_robot(model.current_course.get_objects_by_name('egg')[0].center, obstacle_padding=10, max_curvature=0.05, num_waypoints=100)
    print(f"Planned path with {len(path)} waypoints.")
    model.draw_path(path) # draws the planned path on the current processed frame

    path = model.plan_smooth_path_from_robot(model.current_course.get_objects_by_name('big_goal')[0].center, obstacle_padding=10, max_curvature=0.01, num_waypoints=100)
    print(f"Planned path to large goal with {len(path)} waypoints.")
    model.draw_path(path, color=(0, 255, 0), thickness=3) # draws the planned path to the large goal on the current processed frame

    # generate a path to each white ball
    
    #for ball in model.current_course.get_objects_by_name('white'):
        #path = model.plan_smooth_path_from_robot(ball.center, obstacle_padding=10, max_curvature=0.01, num_waypoints=100)
        #print(f"Planned path to white ball at {ball.center} with {len(path)} waypoints.")
        #model.draw_path(path, color=(255, 0, 0), thickness=2) # draws the planned path to each white ball on the current processed frame

   # path = model.plan_smooth_path(model.current_course.get_objects_by_name('big_goal')[0].center, model.current_course.get_objects_by_name('small_goal')[0].center ,obstacle_padding=10, max_curvature=0.01, num_waypoints=100)
   # model.draw_path(path, color=(12, 120, 255), thickness=2) # draws the planned path from large goal to small goal on the current processed frame
    #path = model.plan_smooth_path(model.current_course.get_objects_by_name('big_goal')[0].center, model.current_course.get_objects_by_name('egg')[0].center ,obstacle_padding=10, max_curvature=0.06, num_waypoints=100)
    #model.draw_path(path, color=(0, 255, 255), thickness=2) # draws the planned path from small goal to large goal on the current processed frame

    cv2.imshow("Processed Frame", model.current_processed_drawn_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def demo2():
    model = AIModel()

    model.set_options(show_boxes=False, show_masks=True, show_confidence=False, show_labels=False, show_center=False)
    model.set_excluded_classes(['wall'])

    img2 = cv2.imread("AI/images/image_26.jpg")
    model.process_frame(img2) # processes the second image
    model.draw_results() # draws the results on the second processed frame

    overlapping_balls = model.get_balls_overlapping_with_robot() # finds the balls overlapping with the robot in the second processed frame
    if overlapping_balls:
        print(f"Found {len(overlapping_balls)} balls overlapping with the robot in the second frame.")
        for ball in overlapping_balls:
            print(ball)
            model.highlight_ball(ball) # highlights the overlapping balls in the second processed frame
    else:
        print("No balls overlapping with the robot in the second frame.")
    
    cv2.imshow("Processed Frame 2", model.current_processed_drawn_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def demo3():
    model = AIModel()

    model.set_options(show_boxes=True, show_masks=False, show_confidence=False, show_labels=False, show_center=False)
    model.set_excluded_classes(['wall'])

    img3 = cv2.imread("AI/images/image_375.jpg")
    model.process_frame(img3) # processes the third image
    model.draw_results() # draws the results on the third processed frame

    direction = model.determine_robot_direction() # determines the direction of the robot in the third processed frame
    model.draw_robot_direction() # draws the direction of the robot on the current processed frame

    print(f"Robot direction angle: {direction} degrees")

    cv2.imshow("Processed Frame 3", model.current_processed_drawn_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def demo4():
    model = AIModel()

    model.set_options(show_boxes=True, show_masks=False, show_confidence=False, show_labels=True, show_center=False)
    model.set_excluded_classes(['wall'])

    img4 = cv2.imread("AI/images/image_432.jpg")
    model.process_frame(img4)
    model.draw_results()
    model.draw_robot_direction()
    closest_ball = model.find_closest_ball("white")
    if closest_ball:
        model.highlight_ball(closest_ball)
        print(f"Closest white ball: {closest_ball}")
        path = model.plan_smooth_path_from_robot(
            closest_ball.center, 
            obstacle_padding=10, 
            max_curvature=0.05, 
            num_waypoints=100
        )
        model.draw_path(path, color=(255, 0, 0), thickness=2)  # Red path
    else:
        print("No white ball found.")
    cv2.imshow("Processed Frame 4", model.current_processed_drawn_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def grid_demo():
    model = AIModel()

    model.set_options(show_boxes=True, show_masks=False, show_confidence=False, show_labels=True, show_center=False)

    img4 = cv2.imread("AI/images/image_86.jpg")
    model.process_frame(img4) # processes the fourth image
    model.draw_results() # draws the results on the fourth processed frame

    grid = model.create_grid_from_image(cell_size=1) # creates a grid from the current processed frame
    print("Grid representation:")
    print(grid) # prints the grid representation
    grid_image = model.convert_grid_to_grid_image(grid) # converts the grid to a color image for visualization

    cv2.imshow("Processed Frame 4", model.current_processed_drawn_frame) # displays the grid image
    cv2.imshow("Grid Image", grid_image) # displays the grid image
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

def dijkstra_demo_fixed():
    model = AIModel()
    
    # Set up visualization options
    model.set_options(show_boxes=True, show_masks=False, show_confidence=False, show_labels=True, show_center=False)
    model.set_excluded_classes(['wall'])  # Exclude walls from visualization but they'll still be obstacles
    
    # Load and process the image
    img = cv2.imread("AI/images/image_432.jpg")
    if img is None:
        print("Error: Could not load image. Check the file path.")
        return
    
    print("Processing frame...")
    model.process_frame(img)
    model.draw_results()
    
    # Debug: Print what objects were found
    print("\nObjects found:")
    for obj_name, count in model.current_course.object_count.items():
        print(f"  {obj_name}: {count}")
    
    # Check if robot exists
    robot_objects = model.current_course.get_objects_by_name("robot")
    if not robot_objects:
        print("WARNING: No robot found!")
        return
    
    # Draw robot direction if possible
    try:
        model.draw_robot_direction()
    except Exception as e:
        print(f"Could not draw robot direction: {e}")
    
    # Use the fixed pathfinding method
    print("\nCalculating distances and paths...")
    
    try:
        cell_size = 20  # Larger cell size for easier pathfinding
        model.draw_dijkstra_paths(cell_size=cell_size)
    except Exception as e:
        print(f"Error in pathfinding: {e}")
        import traceback
        traceback.print_exc()
    
    # Show results
    cv2.imshow("Fixed Pathfinding Demo", model.current_processed_drawn_frame)
    print("\nPress any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    dijkstra_demo_fixed()