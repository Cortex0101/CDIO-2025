import cv2
import numpy as np

# Initial HSV range values for white ball
# These values will be dynamically adjusted from trackbars
h_min_wb, s_min_wb, v_min_wb = 85, 0, 232
h_max_wb, s_max_wb, v_max_wb = 170, 160, 255


# Initial HSV range values for orange ball
# These values will be dynamically adjusted from trackbars
h_min_0b, s_min_ob, v_min_ob = 85, 0, 232
h_max_ob, s_max_ob, v_max_ob = 170, 160, 255

# Callback function for trackbars (does nothing but needed for trackbars)
def on_trackbar(val):
    pass

# Open the default camera
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
if not cap.isOpened():
    print("Error: Camera not accessible!")
    exit()



# Create a window for trackbars
cv2.namedWindow("HSV Trackbars")


# Create trackbars to adjust HSV values dynamically
cv2.createTrackbar("H Min", "HSV Trackbars", h_min_wb, 179, on_trackbar)
cv2.createTrackbar("H Max", "HSV Trackbars", h_max_wb, 179, on_trackbar)
cv2.createTrackbar("S Min", "HSV Trackbars", s_min_wb, 255, on_trackbar)
cv2.createTrackbar("S Max", "HSV Trackbars", s_max_wb, 255, on_trackbar)
cv2.createTrackbar("V Min", "HSV Trackbars", v_min_wb, 255, on_trackbar)
cv2.createTrackbar("V Max", "HSV Trackbars", v_max_wb, 255, on_trackbar)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Empty frame!")
        break

    # Convert frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Get the dynamically adjusted HSV values from trackbars
    h_min_wb = cv2.getTrackbarPos("H Min", "HSV Trackbars")
    h_max_wb = cv2.getTrackbarPos("H Max", "HSV Trackbars")
    s_min_wb = cv2.getTrackbarPos("S Min", "HSV Trackbars")
    s_max_wb = cv2.getTrackbarPos("S Max", "HSV Trackbars")
    v_min_wb = cv2.getTrackbarPos("V Min", "HSV Trackbars")
    v_max_wb = cv2.getTrackbarPos("V Max", "HSV Trackbars")

    lower_color = np.array([h_min_wb, s_min_wb, v_min_wb])
    upper_color = np.array([h_max_wb, s_max_wb, v_max_wb])

    # Apply threshold to detect the selected color
    mask = cv2.inRange(hsv, lower_color, upper_color)

    # Find contours of detected objects
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 180 and area < 230:  # Ignore small and big objects
            M = cv2.moments(contour)
            if M["m00"] != 0:  # Prevent division by zero
                cx = int(M["m10"] / M["m00"])  # X coordinate of center
                cy = int(M["m01"] / M["m00"])  # Y coordinate of center

                print(f"Object Center: ({cx}, {cy})")

                # Draw bounding box
                bounding_box = cv2.boundingRect(contour)
                x, y, w, h = bounding_box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Draw center point
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

                # Display coordinates on the frame
                text = f"X: {cx} Y: {cy}"
                cv2.putText(frame, text, (cx + 10, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Show the output frames
    cv2.imshow("Original Frame", frame)
    cv2.imshow("Thresholded Mask", mask)

    # Press 'Esc' to exit
    if cv2.waitKey(30) == 27:
        break

cap.release()
cv2.destroyAllWindows()


    