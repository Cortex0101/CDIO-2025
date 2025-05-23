import numpy as np
import cv2 as cv
import time

cap = cv.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
# Capture frame-by-frame
    ret, frame = cap.read()
    cv.imwrite('./frame.jpg', frame)

    time.sleep(1)

# if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
# Our operations on the frame come here
    gray = cv.cvtColor(frame, cv.INTER_NEAREST)
# Display the resulting frame
#    cv.imshow('frame', gray)
    if cv.waitKey(1) == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()

