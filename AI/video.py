from ultralytics import YOLO
import cv2
import numpy as np
from sklearn.decomposition import PCA

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



# Load your trained model
#model = YOLO("ball_detect/v3_balls_s_night_run/weights/best.pt")
model = YOLO("ball_detect/v7dtu4/weights/best.pt")
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

    # Loop over each detected box
    for box in r.boxes:
        # xyxy = [[x1, y1, x2, y2]]
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        conf = box.conf[0].cpu().item()

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
