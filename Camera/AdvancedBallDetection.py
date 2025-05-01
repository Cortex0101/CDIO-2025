import cv2
import numpy as np

DEBUGGING = True  # Set to False to disable debugging visuals

def adjust_gamma(frame, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** invGamma * 255
                      for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(frame, table)

def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    equalized = clahe.apply(gray)
    gamma_corrected = adjust_gamma(equalized, gamma=1.5)
    blurred = cv2.GaussianBlur(gamma_corrected, (9, 9), 0)
    return blurred

def dynamic_brightness_calibration(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, maxVal, _, _ = cv2.minMaxLoc(gray)
    scaling_factor = 255.0 / maxVal
    calibrated = cv2.convertScaleAbs(frame, alpha=scaling_factor, beta=0)
    return calibrated

def generate_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Dynamic HSV threshold based on brightness
    _, maxVal, _, _ = cv2.minMaxLoc(hsv[:, :, 2])
    lower_white_hsv = np.array([0, 0, max(0, maxVal - 50)], dtype=np.uint8)
    upper_white_hsv = np.array([180, 50, 255], dtype=np.uint8)
    mask_hsv = cv2.inRange(hsv, lower_white_hsv, upper_white_hsv)

    lower_rgb_white = np.array([200, 200, 200])
    upper_rgb_white = np.array([255, 255, 255])
    mask_rgb = cv2.inRange(rgb, lower_rgb_white, upper_rgb_white)

    combined_mask = cv2.bitwise_and(mask_hsv, mask_rgb)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask_open = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask_closed = cv2.morphologyEx(mask_open, cv2.MORPH_CLOSE, kernel, iterations=2)

    return mask_closed

def detect_circles(mask):
    circles = cv2.HoughCircles(
        mask, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
        param1=50, param2=15,
        minRadius=5, maxRadius=50
    )

    ball_coords = []
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for circle in circles[0, :]:
            ball_coords.append((int(circle[0]), int(circle[1])))

    return ball_coords, circles

def detect_balls(frame, DEBUGGING=False):
    calibrated_frame = dynamic_brightness_calibration(frame)
    preprocessed = preprocess_frame(calibrated_frame)
    mask = generate_mask(calibrated_frame)
    ball_coords, circles = detect_circles(mask)

    if DEBUGGING:
        debug_frame = calibrated_frame.copy()
        if circles is not None:
            for circle in circles[0, :]:
                cv2.circle(debug_frame, (circle[0], circle[1]), circle[2], (0, 255, 0), 2)
                cv2.circle(debug_frame, (circle[0], circle[1]), 2, (0, 0, 255), 3)

        frames = [
            ("Original Frame", frame),
            ("Calibrated Frame", calibrated_frame),
            ("Preprocessed Frame", preprocessed),
            ("Mask", cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)),  # Convert mask to color for text
            ("Detected Balls", debug_frame)
        ]

        current_frame = 0
        while True:
            window_name, frame_to_show = frames[current_frame]
            frame_with_label = frame_to_show.copy()

            # Add label text to the top-right corner
            label = window_name
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame_with_label, (frame_with_label.shape[1] - w - 20, 10),
                          (frame_with_label.shape[1] - 10, h + 20), (0, 0, 0), -1)
            cv2.putText(frame_with_label, label, (frame_with_label.shape[1] - w - 15, h + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("Debug View", frame_with_label)

            key = cv2.waitKey(0) & 0xFF

            if key == ord('d'):
                current_frame = (current_frame + 1) % len(frames)
            elif key == ord('a'):
                current_frame = (current_frame - 1) % len(frames)
            elif key == ord('q'):
                cv2.destroyAllWindows()
                return None

    return ball_coords



# Example usage
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)  # Change as needed

    while True:
        ball_positions = detect_balls(cap)
        print("Detected balls:", ball_positions)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
