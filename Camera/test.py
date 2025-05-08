import cv2
import numpy as np

num = 0
# take image from camera and save it to a file neg_num.jpg
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

# show the camera feed in a window
# when space is pressed, take a picture and save it to a file
cv2.namedWindow("Camera Feed")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to read frame from camera.")
        break

    cv2.imshow("Camera Feed", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' '):
        filename = f"neg_{num}.jpg"
        cv2.imwrite("D:\\CDIO25\\CDIO-2025\\ressources\\img\\positive\\" + filename, frame) 
        print(f"Saved {filename}")
        num += 1
