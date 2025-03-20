import cv2 as cv
import numpy as np

# Start video capture
cap = cv.VideoCapture(0, cv.CAP_DSHOW)

while True:
    # Læs en frame fra kameraet
    ret, frame = cap.read()
    if not ret:
        print("Kan ikke hente frame. Afslutter...")
        break
    
    # Konverter til gråtoner og slør for at reducere støj
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (5, 5), 0)

    # Brug Canny-kantdetektering
    edges = cv.Canny(blurred, 50, 150)

    # Find konturer i kanten
    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # Approximér hver kontur
        perimeter = cv.arcLength(contour, True)
        approx = cv.approxPolyDP(contour, 0.02 * perimeter, True)

        # Kontroller for rektangulært formet objekt
        if len(approx) >= 12:  # Antag at det kræver mange linjer i det detaljerede kors
            # Først tjek størrelsen, så små elementer udelukkes
            area = cv.contourArea(contour)
            if area > 100:  # Kan justeres afhængig af det forventede kors' størrelse
                # Brug boundingRect for yderligere inspektion
                x, y, w, h = cv.boundingRect(approx)
                aspectRatio = float(w) / h
                if 0.8 <= aspectRatio <= 1.2:  # Tjek for aspektsforhold tæt på en firkant
                    cv.drawContours(frame, [approx], -1, (0, 255, 0), 2)
                    cv.putText(frame, "Cross Detected", (x, y - 10),
                               cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv.imshow("Frame with Cross Detection", frame)

    # Tryk 'q' for at afslutte
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
