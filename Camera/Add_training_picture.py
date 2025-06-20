import os
import cv2 as cv
import numpy as np

# Start video capture
cap = cv.VideoCapture(0, cv.CAP_DSHOW)
# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()
# Set the camera resolution to 640x480
cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)   
ret, frame = cap.read()
if not ret:
    print("Error: Could not read frame.")
    exit()
# enter folder path

folder = input("Enter folder: ")
while True:
 filename = input("Enter file name (without extension) or 'q' to quit: ")

 if cv.waitKey(1) & 0xFF == ord("q"):
        break

 full_path = os.path.join(folder, filename + ".jpg")

 cv.imwrite(full_path, frame)
 print(f"First picture saved as {full_path}.")
