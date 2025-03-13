import cv2
import numpy as np

# Initial HSV range values for white ball
h_min_wb, s_min_wb, v_min_wb = 10, 0, 231
h_max_wb, s_max_wb, v_max_wb = 179, 14, 255

# Initial HSV range values for orange ball
h_min_ob, s_min_ob, v_min_ob = 0, 0, 231
h_max_ob, s_max_ob, v_max_ob = 40, 212, 255

#h_min_ob, s_min_ob, v_min_ob = 0, 177, 182
#h_max_ob, s_max_ob, v_max_ob = 13, 243, 212

# Initial HSV range values for obstacles
h_min_obsta, s_min_obsta, v_min_obsta = 167, 137, 206
h_max_obsta, s_max_obsta, v_max_obsta = 179, 255, 255

# Lists to store detected balls
white_balls = []
orange_balls = []

# Obstacle detection
obstacle_x, obstacle_y = 0, 0
obstacle = []

def doesBallExistInList(ballList, x, y):
    for ball in ballList:
        if abs (ball[0] - x) < 10 and abs (ball[1] - y) < 10:
            return True
    return False

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

# Clear the list of detected white balls
white_balls.clear()

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
        if area > 10 and area < 300:  # Ignore small and big objects
            M = cv2.moments(contour)
            if M["m00"] != 0:  # Prevent division by zero
                cx = int(M["m10"] / M["m00"])  # X coordinate of center
                cy = int(M["m01"] / M["m00"])  # Y coordinate of center

                if not doesBallExistInList(white_balls, cx, cy):
                 white_balls.append((cx, cy))

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

# cap.release()
cv2.destroyAllWindows()


# Create a window for trackbars
cv2.namedWindow("HSV Trackbars")

# Create trackbars to adjust HSV values dynamically
cv2.createTrackbar("H Min", "HSV Trackbars", h_min_ob, 179, on_trackbar)
cv2.createTrackbar("H Max", "HSV Trackbars", h_max_ob, 179, on_trackbar)
cv2.createTrackbar("S Min", "HSV Trackbars", s_min_ob, 255, on_trackbar)
cv2.createTrackbar("S Max", "HSV Trackbars", s_max_ob, 255, on_trackbar)
cv2.createTrackbar("V Min", "HSV Trackbars", v_min_ob, 255, on_trackbar)
cv2.createTrackbar("V Max", "HSV Trackbars", v_max_ob, 255, on_trackbar)


# Clear the list of detected orange balls
orange_balls.clear()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Empty frame!")
        break

    # Convert frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Get the dynamically adjusted HSV values from trackbars
    h_min_ob = cv2.getTrackbarPos("H Min", "HSV Trackbars")
    h_max_ob = cv2.getTrackbarPos("H Max", "HSV Trackbars")
    s_min_ob = cv2.getTrackbarPos("S Min", "HSV Trackbars")
    s_max_ob = cv2.getTrackbarPos("S Max", "HSV Trackbars")
    v_min_ob = cv2.getTrackbarPos("V Min", "HSV Trackbars")
    v_max_ob = cv2.getTrackbarPos("V Max", "HSV Trackbars")

    lower_color = np.array([h_min_ob, s_min_ob, v_min_ob])
    upper_color = np.array([h_max_ob, s_max_ob, v_max_ob])

    # Apply threshold to detect the selected color
    mask = cv2.inRange(hsv, lower_color, upper_color)


    # Find contours of detected objects
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 120 and area < 300:  # Ignore small and big objects
            M = cv2.moments(contour)
            if M["m00"] != 0:  # Prevent division by zero
                cx = int(M["m10"] / M["m00"])  # X coordinate of center
                cy = int(M["m01"] / M["m00"])  # Y coordinate of center

                if not doesBallExistInList(orange_balls, cx, cy):
                    orange_balls.append((cx, cy))

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

#cap.release()
cv2.destroyAllWindows()




# Create a window for trackbars
cv2.namedWindow("HSV Trackbars")

# Create trackbars to adjust HSV values dynamically
cv2.createTrackbar("H Min", "HSV Trackbars", h_min_obsta, 179, on_trackbar)
cv2.createTrackbar("H Max", "HSV Trackbars", h_max_obsta, 179, on_trackbar)
cv2.createTrackbar("S Min", "HSV Trackbars", s_min_obsta, 255, on_trackbar)
cv2.createTrackbar("S Max", "HSV Trackbars", s_max_obsta, 255, on_trackbar)
cv2.createTrackbar("V Min", "HSV Trackbars", v_min_obsta, 255, on_trackbar)
cv2.createTrackbar("V Max", "HSV Trackbars", v_max_obsta, 255, on_trackbar)


# Clear the list of detected obstacles
obstacle.clear()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Empty frame!")
        break

    # Convert frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Get the dynamically adjusted HSV values from trackbars
    h_min_obsta = cv2.getTrackbarPos("H Min", "HSV Trackbars")
    h_max_obsta = cv2.getTrackbarPos("H Max", "HSV Trackbars")
    s_min_obsta = cv2.getTrackbarPos("S Min", "HSV Trackbars")
    s_max_obsta = cv2.getTrackbarPos("S Max", "HSV Trackbars")
    v_min_obsta = cv2.getTrackbarPos("V Min", "HSV Trackbars")
    v_max_obsta = cv2.getTrackbarPos("V Max", "HSV Trackbars")

    lower_color = np.array([h_min_obsta, s_min_obsta, v_min_obsta])
    upper_color = np.array([h_max_obsta, s_max_obsta, v_max_obsta])

    # Apply threshold to detect the selected color
    mask = cv2.inRange(hsv, lower_color, upper_color)


    # Find contours of detected objects
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000 and area < 5000:  # Ignore small and big objects
            M = cv2.moments(contour)
            if M["m00"] != 0:  # Prevent division by zero
                cx = int(M["m10"] / M["m00"])  # X coordinate of center
                cy = int(M["m01"] / M["m00"])  # Y coordinate of center

                if not doesBallExistInList(obstacle, cx, cy):
                    obstacle.append((cx, cy))

                print(f"Object Center for obstacle: ({cx}, {cy})")

                # Draw bounding box
                bounding_box = cv2.boundingRect(contour)
                x, y, w, h = bounding_box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Draw center point
                #cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                cv2.drawContours(frame, contour, -1, (0, 255, 0), 2)

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




for x,y in white_balls:
    print(f"White Ball Center: ({x}, {y})")

for x,y in orange_balls:
    print(f"Orange Ball Center: ({x}, {y})") 


for x,y in obstacle:
    print(f"Obstacle Center: ({x}, {y})")      

