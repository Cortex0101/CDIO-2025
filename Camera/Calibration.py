import cv2
import numpy as np




# Callback function for trackbars (does nothing but needed for trackbars)
def on_trackbar(val):
    pass

# Create a window for trackbars
cv2.namedWindow("Canny Trackbars")
canny_threshold1 = 39
canny_threshold2 = 100



gaussian_blur_size_x = 11
gaussian_blur_size_y = 11
gaussian_blur_sigma = 0

with open('Calibration.txt', 'r') as file:
    for line in file:
        # Del linjen ved kolon og fjern overskydende mellemrum
        key, value = line.strip().split(':')
        
        # Tildel værdier baseret på nøgle
        if key == 'Canny_threshold1':
            canny_threshold1 = int(value.strip())
        elif key == 'Canny_threshold2':
            canny_threshold2 = int(value.strip())
        elif key == 'Gaussian_blur_sigma':
            gaussian_blur_sigma = int(value.strip())
#print(f"canny_threshold1: {canny_threshold1}")
#print(f"canny_threshold2: {canny_threshold2}")      
#print(f"gaussian_blur_sigma: {gaussian_blur_sigma}")

# Create trackbars to adjust Blur and Canny values dynamically
cv2.createTrackbar("Canny_thr1", "Canny Trackbars", canny_threshold1, 200, on_trackbar)
cv2.createTrackbar("Canny_thr2", "Canny Trackbars", canny_threshold2, 300, on_trackbar)   
cv2.createTrackbar("Gaussian_blur_sigma", "Canny Trackbars", gaussian_blur_sigma, 50, on_trackbar)




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


while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Empty frame!")
        break

    
    # Get the dynamically adjusted HSV values from trackbars
    Canny_threshold1 = cv2.getTrackbarPos("Canny_thr1", "Canny Trackbars")
    Canny_threshold2 = cv2.getTrackbarPos("Canny_thr2", "Canny Trackbars")  
    Gaussian_blur_sigma = cv2.getTrackbarPos("Gaussian_blur_sigma", "Canny Trackbars")

    # Convert frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
   


    # Press 'Esc' to exit
    if cv2.waitKey(30) == 27:
        break
 
    #  konvert frame to Gray tones
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   
    # Detecting goals
    # Konverter til gråtoner og slør for at reducere støj
 #   gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (gaussian_blur_size_x, gaussian_blur_size_y), gaussian_blur_sigma)

    # Brug Canny-kantdetektering
    edges = cv2.Canny(blurred, canny_threshold1, canny_threshold2)

    cv2.imshow("Edges", edges)

    with open("Calibration.txt", "w") as f:
        
        f.write(f"Canny_threshold1: {canny_threshold1}\n")
        f.write(f"Canny_threshold2: {canny_threshold2}\n")    
        f.write(f"Gaussian_blur_sigma: {gaussian_blur_sigma}\n")