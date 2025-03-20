import cv2 as cv
import numpy as np

# Start video capture
cap = cv.VideoCapture(0, cv.CAP_DSHOW)

#cap.set(cv.CAP_PROP_FRAME_WIDTH, 800)
#cap.set(cv.CAP_PROP_FRAME_HEIGHT, 600)
if not cap.isOpened():
    print("Error: Camera not accessible!")
    exit()

while True:
    # Læs en frame fra kameraet
    ret, frame = cap.read()
    if not ret:
        print("Kan ikke hente frame. Afslutter...")
        break

    # Konverter frame til HSV farverum
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # Definer HSV område for orange farve og brug den som maske
    lower_orange = np.array([5, 150, 100])
    upper_orange = np.array([50, 255, 255])
  #  lower_orange = np.array([0, 0, 0])
   # upper_orange = np.array([255, 255, 255])
    mask_orange = cv.inRange(hsv, lower_orange, upper_orange)

    # Brug masken til at finde konturer
    contours, _ = cv.findContours(mask_orange, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # Beregn cirklens omkreds og se om det er en cirkel
        perimeter = cv.arcLength(contour, True)
        approx = cv.approxPolyDP(contour, 0.04 * perimeter, True)

        if len(approx) > 8:  # En tilnærmelse for en cirkel
            ((x, y), radius) = cv.minEnclosingCircle(approx)
            center = (int(x), int(y))
            
            # Tegn cirklen hvis den er stor nok
            if radius > 10:
                cv.circle(frame, center, int(radius), (0, 255, 0), 2)
                cv.putText(frame, "Orange Circle", (int(x - radius), int(y - radius)),
                           cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv.imshow("Detected Orange Circles", frame)

    # Press 'Esc' to exit
    if cv.waitKey(30) == 27:
        break
cap.release()
cv.destroyAllWindows()
