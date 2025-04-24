import cv2
import numpy as np

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

def get_ball_positions():
    ret, frame = cap.read()

    if not ret:
        print("Error: Unable to read frame from camera.")
        return None

    white_balls = []
    orange_balls = []
    obstacle = []
    egg = []

    # Parameters
    Gaussian_blur_size = (11, 11)
    Gaussian_blur_sigma = 0
    canny_threshold1 = 39
    canny_threshold2 = 100

    # Convert to grayscale and blur
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, Gaussian_blur_size, Gaussian_blur_sigma)

    # Edge detection
    edges = cv2.Canny(blurred, canny_threshold1, canny_threshold2)

    # Convert to HSV for color classification
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    def circularity(area, perimeter):
        return 4 * np.pi * (area / (perimeter * perimeter)) if perimeter != 0 else 0

    def is_close(ballList, x, y, tolerance):
        for bx, by in ballList:
            if abs(bx - x) < tolerance and abs(by - y) < tolerance:
                return True
        return False

    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        if perimeter < 10 or perimeter > 500:
            continue

        approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
        ((x, y), radius) = cv2.minEnclosingCircle(approx)
        center = (int(x), int(y))
        circ = circularity(area, perimeter)

        # Detect Balls
        if 0.8 < circ < 1.0 and 6 < radius < 12 and len(approx) > 4 and len(approx) < 8:
            mask = np.zeros_like(hsv)
            cv2.circle(mask, center, int(radius), (255, 255, 255), -1)
            mean_val = cv2.mean(hsv, mask[:,:,0])

            if 70 < mean_val[0] < 140 and 5 < mean_val[1] < 100:
                if not is_close(white_balls, int(x), int(y), 15):
                    white_balls.append((int(x), int(y)))
            elif 10 < mean_val[0] < 50 and 50 < mean_val[1] < 150:
                if not is_close(orange_balls, int(x), int(y), 15):
                    orange_balls.append((int(x), int(y)))
        
        # Detect Egg (larger, slightly elongated circle)
        if 0.8 < circ < 0.9 and 15 < radius < 25:
            if not is_close(egg, int(x), int(y), 15):
                egg.append((int(x), int(y)))

        # Detect Obstacle-like shape (cross with angles)
        if 8 <= len(approx) < 15 and 70 < area < 5500:
            x_box, y_box, w, h = cv2.boundingRect(approx)
            cx = int(x_box + w / 2)
            cy = int(y_box + h / 2)

            if not is_close(obstacle, cx, cy, 50):
                obstacle.append((cx, cy))

    return {
        "white_balls": white_balls,
        "orange_balls": orange_balls,
        "obstacles": obstacle,
        "eggs": egg
    }
