from ultralytics import YOLO
import cv2
import numpy as np
import math

from sklearn.decomposition import PCA


DEBUG = True  # Set to True to enable debug mode

def calculate_robot_orientation(polygon_points):
    mask_coords = np.array(polygon_points, dtype=np.float32)

    # Step 1: PCA
    pca = PCA(n_components=2)
    pca.fit(mask_coords)
    main_axis = pca.components_[0]
    center = pca.mean_

    # Step 2: Project points to split front/back
    main_axis /= np.linalg.norm(main_axis)
    projections = np.dot(mask_coords - center, main_axis)

    front_side = mask_coords[projections > 0.05]
    back_side = mask_coords[projections < -0.05]

    def side_complexity_score(side_pts):
        if len(side_pts) < 3:
            return 0
        hull = cv2.convexHull(side_pts)
        area_hull = cv2.contourArea(hull)
        area_poly = cv2.contourArea(side_pts)
        return max(area_hull - area_poly, 0)

    score_front = side_complexity_score(front_side)
    score_back = side_complexity_score(back_side)

    if score_front >= score_back:
        direction_vector = main_axis
    else:
        direction_vector = -main_axis

    angle_rad = np.arctan2(direction_vector[1], direction_vector[0])
    angle_deg = (np.degrees(angle_rad) + 360) % 360
    return angle_deg, center, direction_vector

def draw_robot_with_direction(image, polygon_points, angle_deg, center, direction_vector, scale=80):
    polygon = np.array(polygon_points, dtype=np.int32)

    # Draw filled polygon outline
    cv2.polylines(image, [polygon], isClosed=True, color=(0, 255, 0), thickness=2)

    # Draw center point
    cx, cy = int(center[0]), int(center[1])
    cv2.circle(image, (cx, cy), 4, (0, 0, 255), -1)

    # Draw orientation arrow
    dx = int(scale * direction_vector[0])
    dy = int(scale * direction_vector[1])
    tip = (cx + dx, cy + dy)
    cv2.arrowedLine(image, (cx, cy), tip, (255, 0, 0), 2, tipLength=0.3)

    # Annotate angle
    cv2.putText(image, f"{angle_deg:.1f} deg", (cx + 10, cy - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return image

def get_robot_angle(frame):
    angle = None

    # green range
    color1_hsv = (np.array([70, 100, 50]), np.array([95, 255, 200]))
    # yellow range
    color2_hsv = (np.array([20, 100, 100]), np.array([35, 255, 255]))

    if not ret:
        print("Error: Unable to read frame from camera.")
        return None
        
    display_frame = frame.copy()  # Create a copy for visualization
    
        
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # makes masks[color]
    masks = {color: cv2.inRange(hsv, *hsv_range) for color, hsv_range in zip(('color1', 'color2'), (color1_hsv, color2_hsv))}
    centers = {}
    
    for color, mask in masks.items():
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            # area size to catch, currently: 100 pixels
            if cv2.contourArea(largest) > 1:
                M = cv2.moments(largest)
                centers[color] = (int(M['m10']/M['m00']), int(M['m01']/M['m00']))

                    # Draw the contour and center for each color
                color_bgr = (0, 0, 255) if color == 'color1' else (0, 255, 255)  # Red for color1, Yellow for color2
                cv2.drawContours(display_frame, [largest], -1, color_bgr, 2)
                cv2.circle(display_frame, centers[color], 5, color_bgr, -1)
    
    result = np.zeros_like(frame)


    if 'color1' in centers and 'color2' in centers:

        # Draw line between the two color centers
        cv2.line(display_frame, centers['color1'], centers['color2'], (255, 255, 255), 2)
        
        # Calculate angle
        dx, dy = np.subtract(centers['color1'], centers['color2'])
        angle = (math.degrees(math.atan2(dy, dx)) + 360) % 360
        
        # Add text showing the angle
        text_pos = ((centers['color1'][0] + centers['color2'][0]) // 2, 
                   (centers['color1'][1] + centers['color2'][1]) // 2 - 20)
        cv2.putText(display_frame, f"Angle: {angle:.1f}°", text_pos, 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Display the frame
    if DEBUG:
        cv2.imshow("Robot Direction", display_frame)
        cv2.waitKey(1)  # Small delay to allow display to update

    return angle



# Load your trained model
#model = YOLO("ball_detect/v3_balls_s_night_run/weights/best.pt")
model = YOLO("ball_detect/v8/weights/best.pt")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run inference (returns a list of Results; we only care about the first)
    results = model.predict(source=frame, conf=0.3, iou=0.5)
    r = results[0]

    """"
    # find "robot" segmentation mask if available
    if r.masks is not None:
        # find label "robot" in the masks
        robot_mask = None
        for i, mask in enumerate(r.masks):
            if r.names[mask.cls[0].item()] == "robot":
                robot_mask = mask
                break

        if robot_mask is not None:
            # Convert mask to polygon points
            mask_coords = robot_mask.xy.cpu().numpy().astype(int)
            if len(mask_coords) > 0:
                # Calculate orientation angle
                angle_deg, center, direction_vector = calculate_robot_orientation(mask_coords[0])

                # Draw robot with direction
                frame = draw_robot_with_direction(frame, mask_coords[0], angle_deg, center, direction_vector)
    else:
        print("No robot mask found in the current frame.")
    # If no robot mask, continue with detection boxes
    """

    # Convert to a writable image
    img = frame.copy()

    ang = get_robot_angle(img)
    print(f"Robot angle: {ang} degrees")

    eggs = []    
    crosses = []

    # Loop over each detected box
    for box in r.boxes:
        # xyxy = [[x1, y1, x2, y2]]
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        conf = box.conf[0].cpu().item()

        # pinpoint correct egg/cross
        cls = int(box.cls[0].item())
        area = (x2 - x1) * (y2 - y1)
        if cls == 2: # egg i think? Maybe find by "egg" if it doesn't work
            eggs.append((area, x1, y1, x2, y2, conf))
        elif cls == 3: # cross
            crosses.append((area, x1, y1, x2, y2, conf))
        
        for detections in [crosses, eggs]:
            if detections:
                largest = max(detections, key=lambda x: x[0])
                area, x1, y1, x2, y2, conf = largest

        # compute center
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        # draw rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # show confidence above the box
        cv2.putText(
            img,
            f"{conf:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

        # draw center‐point and coords
        cv2.circle(img, (cx, cy), 3, (0, 0, 255), -1)
        cv2.putText(
            img,
            f"({cx},{cy})",
            (cx + 5, cy - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
        )

    # display
    cv2.imshow("Detections + Coords", img)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
