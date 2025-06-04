"""
# label_image.py
Allows you to load an image and display it
You can then draw a rectangle on the image by clicking the top left corner and the bottom right corner of the rectangle
The rectangle will be drawn on the image and the coordinates of the rectangle will be printed in the console
The rectangle will be saved in a file called "label.txt" in the same directory as the image
"""

import cv2
import numpy as np
import os
import sys
import glob
import random

# Path to the image
img_path = "images/image_76.jpg" # Change this to your image path

# Path to the label file
label_path = "images/" + os.path.basename(img_path).replace(".jpg", ".txt") # Change this to your label path

# Create the label file if it doesn't exist
if not os.path.exists(label_path):
    with open(label_path, "w") as f:
        pass

# Load the image
img = cv2.imread(img_path)
if img is None:
    print("Could not read the image.")
    sys.exit()


img = cv2.resize(img, (640, 640))
img_copy = img.copy()
cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Image", 640, 640)

rectangles = [] # contains top left and width and height of the rectangle

drawing = False # True if the mouse is pressed
start_point = None # top left corner of the rectangle
end_point = None # bottom right corner of the rectangle

current_object_type = 0 # 0 for white ball, 1 orange ball, 2 for cross, 3 for wall, 4 for robot

def draw_rectangle(event, x, y, flags, param):
    global start_point, end_point, drawing, img_copy, rectangles

    if event == cv2.EVENT_LBUTTONDOWN:
        # Start drawing the rectangle
        drawing = True
        start_point = (x, y)
        end_point = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            # Update the end point of the rectangle
            end_point = (x, y)
            img_copy = img.copy()
            cv2.rectangle(img_copy, start_point, end_point, (0, 255, 0), 2)
            cv2.imshow("Image", img_copy)

    elif event == cv2.EVENT_LBUTTONUP:
        # Stop drawing the rectangle
        drawing = False
        end_point = (x, y)

        cv2.rectangle(img_copy, start_point, end_point, (0, 255, 0), 2)
        rectangles.append((start_point[0], start_point[1], end_point[0] - start_point[0], end_point[1] - start_point[1], current_object_type))

        print(f"Rectangle: {start_point} to {end_point}")
        print(f"Width: {end_point[0] - start_point[0]}, Height: {end_point[1] - start_point[1]}")
        print(f"Rectangles: {rectangles}")
        # Save the rectangles to the label file
        with open(label_path, "w") as f:
            for rect in rectangles:
                # find center x y between start and end point
                # rect = (x, y, width, height, object_type)

                center_coord = (rect[0] + rect[2] / 2, rect[1] + rect[3] / 2)
                # convert to yolo width the coord being 0-1 bsed on the image size
                img_height, img_width = img.shape[:2]
                center_x = center_coord[0] / img_width
                center_y = center_coord[1] / img_height

                # convert to yolo width and height
                width = rect[2] / img_width
                height = rect[3] / img_height

                f.write(f"{rect[4]} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")

# Bind the mouse callback function to the window
cv2.setMouseCallback("Image", draw_rectangle)

while True:
    cv2.putText(img_copy, f"Current object type: {current_object_type}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Image", img_copy)

    # Wait for a key press
    key = cv2.waitKey(1) & 0xFF

    # If the 'q' key is pressed, exit the loop
    if key == ord('q'):
        break

    # If the 'r' key is pressed, reset the rectangles
    if key == ord('r'):
        rectangles = []
        img_copy = img.copy()
        print("Rectangles reset")

    # if the z key is pressed, remove the last rectangle
    if key == ord('z'):
        if rectangles:
            rectangles.pop()
            img_copy = img.copy()
            for rect in rectangles:
                cv2.rectangle(img_copy, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 2)
            print("Last rectangle removed")
        else:
            print("No rectangles to remove")

    # if 1 or 2 is pressed, change the current object type
    if key == ord('1'):
        current_object_type = 0
        print("Current object type: Orange ball")
        img_copy = img.copy()
        for rect in rectangles:
            cv2.rectangle(img_copy, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 2)
    elif key == ord('2'):
        current_object_type = 1
        print("Current object type: White ball")
        img_copy = img.copy()
        for rect in rectangles:
            cv2.rectangle(img_copy, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 2)
    elif key == ord('3'):
        current_object_type = 2
        print("Current object type: Egg")
        img_copy = img.copy()
        for rect in rectangles:
            cv2.rectangle(img_copy, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 2)
    elif key == ord('4'):
        current_object_type = 3
        print("Current object type: Cross")
        img_copy = img.copy()
        for rect in rectangles:
            cv2.rectangle(img_copy, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 2)
    elif key == ord('5'):
        current_object_type = 4
        print("Current object type: Robot")
        img_copy = img.copy()
        for rect in rectangles:
            cv2.rectangle(img_copy, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 2)
    elif key == ord('6'):
        current_object_type = 5
        print("Current object type: Small Goal")
        img_copy = img.copy()
        for rect in rectangles:
            cv2.rectangle(img_copy, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 2)
    elif key == ord('7'):
        current_object_type = 6
        print("Current object type: Big Goal")
        img_copy = img.copy()
        for rect in rectangles:
            cv2.rectangle(img_copy, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 2)
    elif key == ord('8'):
        current_object_type = 7
        print("Current object type: Walls")
        img_copy = img.copy()
        for rect in rectangles:
            cv2.rectangle(img_copy, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 2)

    # draw the rectangles on the image
    for rect in rectangles:
        cv2.rectangle(img_copy, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 2)