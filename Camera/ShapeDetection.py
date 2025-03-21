import cv2
import numpy as np


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

white_balls.clear()
orange_balls.clear()
obstacle.clear()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Empty frame!")
        break

    

    # Convert frame to HSV color space
   # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
   


    # Press 'Esc' to exit
    if cv2.waitKey(30) == 27:
        break
 
    # Forbehandling: konvertere frame til gråtoner og HSV
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Reducer støj
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)

    # Udfør Canny kantdetektion
    edges = cv2.Canny(blurred, 30, 150)

    # Find konturer
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # Tilnærm polygonens omkreds
        perimeter = cv2.arcLength(contour, True)
        #For tetecting balls
        approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
        #print ( "approx: \n", approx)
        
        # Kontroller om konturen er en cirkel
        if len(approx) > 5:  # En tilnærmelse for en cirkel
            ((x, y), radius) = cv2.minEnclosingCircle(approx)
            center = (int(x), int(y))
           # print ( "center: \n", center)
           # print ( "radius: \n", radius)
            
            # Undgå at overveje meget små former
            if radius > 5 and radius < 10:
                mask = np.zeros_like(hsv)
                #print ( "mask: \n", mask)
                cv2.circle(mask, center, int(radius), (255, 255, 255), -1)
                masked_hsv = cv2.bitwise_and(hsv, mask)
                #print ( "masked_hsv: \n", masked_hsv)
                cv2.circle(frame, center, 5, (0, 0, 255), -1)
                
                # Draw bounding box
                bounding_box = cv2.boundingRect(contour)
                x, y, w, h = bounding_box
                cv2.rectangle(frame, center, (x + w, y + h), (0, 255, 0), 2)


                mean_val = cv2.mean(hsv, mask[:,:,0])
                #print ( "mean_val: \n", mean_val)
                
                if mean_val[1] < 140:
                    color = "White"
                elif (5 < mean_val[0] < 40) and (150 < mean_val[1] < 255):
                    color = "Orange"
                else:
                    continue

                cv2.circle(frame, center, int(radius), (0, 255, 0), 2)
                cv2.putText(frame, f"{color} Circle", (int(x - radius), int(y - radius)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Display coordinates on the frame
                text = f"X: {x} Y: {y}"
                cv2.putText(frame, text, (x + 10, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                #cv2.imshow("Mask", mask)
                #cv2.imshow("Masked HSV", masked_hsv)

                if not doesBallExistInList(white_balls, x, y) and mean_val[1] < 140:
                    white_balls.append((x, y))
                if not doesBallExistInList(orange_balls, x, y) and (5 < mean_val[0] < 40) and (150 < mean_val[1] < 255):
                    orange_balls.append((x, y))
   #For detecting obstacles
  # for contour in contours:
         # Approximér hver kontur
   #     perimeter = cv2.arcLength(contour, True)
    #    approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        # Kontroller for rektangulært formet objekt
     #   if len(approx) >= 12:  # Antag at det kræver mange linjer i det detaljerede kors
            #print ( "len(approx): \n", len(approx))
            # Først tjek størrelsen, så små elementer udelukkes
    #        area = cv2.contourArea(contour)
     #      print ( "area: \n", area)
     #       if area > 100:  # Kan justeres afhængig af det forventede kors' størrelse
                # Brug boundingRect for yderligere inspektion
    #            x, y, w, h = cv2.boundingRect(approx)
      #          aspectRatio = float(w) / h
      #          print ( "aspectRatio: \n", aspectRatio)
      #          if 0.8 <= aspectRatio <= 1.2:  # Tjek for aspektsforhold tæt på en firkant
      #              cv2.drawContours(frame, [approx], -1, (0, 255, 0), 2)
      #              cv2.putText(frame, "Cross Detected", (x, y - 10),
      #                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

       #             cv2.imshow("Frame with Cross Detection", frame)


    cv2.imshow("Detected circle", frame)
  #  cv2.imshow("Edges", edges)
 #   cv2.imshow("Blurred", blurred)
 #   cv2.imshow("Gray", gray)
 #   cv2.imshow("HSV", hsv)
 #   cv2.imshow("Mask", mask)
#    cv2.imshow("Masked HSV", masked_hsv)
    



cap.release()
cv2.destroyAllWindows()




for x,y in white_balls:
    print(f"White Ball Center: ({x}, {y})")

for x,y in orange_balls:
    print(f"Orange Ball Center: ({x}, {y})") 


for x,y in obstacle:
    print(f"Obstacle Center: ({x}, {y})")      

