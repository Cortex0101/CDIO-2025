import cv2
import numpy as np
import math


# Lists to store detected balls
white_balls = []
orange_balls = []

# Obstacle detection
obstacle_x, obstacle_y = 0, 0
obstacle = []

egg = []
def detect_direction(frame, color1_hsv, color2_hsv):

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # makes masks[color]
    masks = {color: cv2.inRange(hsv, *hsv_range) for color, hsv_range in zip(('color1', 'color2'), (color1_hsv, color2_hsv))}
    centers = {}
    
    angle = None
    
    for color, mask in masks.items():
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            # area size to catch, currently: 10 pixels
            if cv2.contourArea(largest) > 10:
                M = cv2.moments(largest)
                centers[color] = (int(M['m10']/M['m00']), int(M['m01']/M['m00']))
    
    if 'color1' in centers and 'color2' in centers:
        dx, dy = np.subtract(centers['color1'], centers['color2'])
        # change dy, dx if e.g. 90 degrees is now up instead of right etc.
        angle = (math.degrees(math.atan2(dy, dx)) + 360) % 360
    
    return angle
def doesBallExistInList(ballList, x, y):
    for ball in ballList:
        if abs (ball[0] - x) < 15 and abs (ball[1] - y) < 15:
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
egg.clear()

# choose colors for detection of front/rear
# green range
color1_hsv = (np.array([70, 100, 50]), np.array([95, 255, 200]))
# yellow range
color2_hsv = (np.array([20, 100, 100]), np.array([35, 255, 255]))

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Empty frame!")
        break

    

    # Convert frame to HSV color space
    # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
   
    # detection shit down here
    angle = detect_direction(frame, color1_hsv, color2_hsv)

    if angle is not None:
        print(f"Direction: {angle:.0f}")

    # Press 'Esc' to exit
    if cv2.waitKey(30) == 27:
        break
 
    #  konvert frame to Gray tones and HSV
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Reduce noise
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    #blurred = cv2.GaussianBlur(gray, (11, 11), 2)

    # Do Canny edgedetektion
    edges = cv2.Canny(blurred, 30, 150)
    #edges = cv2.Canny(blurred, 50, 220)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # Tilnærm polygonens omkreds
        perimeter = cv2.arcLength(contour, True)
        #For detecting balls
        approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
        #print ( "approx: \n", approx)
        
        # Check if the kocontour is circle
        if len(approx) > 5 and len(approx) < 8 :  # approximation of a circle
            ((x, y), radius) = cv2.minEnclosingCircle(approx)
            center = (int(x), int(y))
           # print ( "center: \n", center)
            #print ( "radius: \n", radius)
            
            # avoid small detections
            if radius > 8 and radius < 12:
                mask = np.zeros_like(hsv)
                #print ( "mask: \n", mask)
                cv2.circle(mask, center, int(radius), (255, 255, 255), -1)
                masked_hsv = cv2.bitwise_and(hsv, mask)
                #print ( "masked_hsv: \n", masked_hsv)
                #cv2.circle(frame, center, 5, (0, 0, 255), -1)
                
                # Draw bounding box
                #bounding_box = cv2.boundingRect(contour)
                #x, y, w, h = bounding_box
                #cv2.rectangle(frame, center, (x + w, y + h), (0, 255, 0), 2)


                mean_val = cv2.mean(hsv, mask[:,:,0])
                #print ( "mean_val: \n", mean_val)
                
                #if mean_val[1] < 140:
                if mean_val[1] < 120:
                #if  mean_val[0] > 210  and mean_val[1] < 150 :
                    color = "White"
                    #print ( "White mean_val: \n", mean_val)
                #elif (5 < mean_val[0] < 40) and (150 < mean_val[1] < 255):
                elif (10 < mean_val[0] < 50) and (50 < mean_val[1] < 210):
                    color = "Orange"
                    #print ( "mean_val: \n", mean_val)
                else:
                    print ( "undefined mean_val: \n", mean_val)
                    continue

                if color == "White" or color == "Orange":
                  # Draw bounding box
                  bounding_box = cv2.boundingRect(contour)
                  x, y, w, h = bounding_box
                  start_rect = (int(x - 50), int(y - 25))
                  cv2.rectangle(frame, start_rect, (x + 100, y + 50), (0, 255, 0), 2)
                  cv2.circle(frame, center, int(2 * radius), (0, 255, 0), 2)
                  cv2.putText(frame, f"{color} Ball", (int(x - 35), int(y + 40)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                  # Display coordinates on the frame
                  text = f"X: {x} Y: {y}"
                  cv2.putText(frame, text, (x - 30, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                
                  if not doesBallExistInList(white_balls, x, y) and color == "White":
                    white_balls.append((x, y))
                  if not doesBallExistInList(orange_balls, x, y) and color == "Orange":
                    orange_balls.append((x, y))
             # Detect egg
            if radius > 20 and radius < 25:
                #print ( "Egg detected. \n")
                mask = np.zeros_like(hsv)
                #print ( "mask: \n", mask)
                cv2.circle(mask, center, int(radius), (255, 255, 255), -1)
                #masked_hsv = cv2.bitwise_and(hsv, mask)
                #mean_val = cv2.mean(hsv, mask[:,:,0])
                bounding_box = cv2.boundingRect(contour)
                x, y, w, h = bounding_box
                start_rect = (int(x - 50), int(y - 25))
                cv2.rectangle(frame, start_rect, (x + 100, y + 50), (0, 255, 0), 2)
                cv2.circle(frame, center, int(2 * radius), (0, 255, 0), 2)
                cv2.putText(frame, f"Egg", (int(x - 35), int(y + 40)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Display coordinates on the frame
                text = f"X: {x} Y: {y}"
                cv2.putText(frame, text, (x - 30, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                if not doesBallExistInList(egg, x, y): egg.append((x, y))


   #For detecting obstacles
  
         # Approximér hver kontur
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
         #Kontroller for rektangulært formet objekt
        if len(approx) >= 12:  # Antag at det kræver mange linjer i det detaljerede kors
            #print ( "len(approx): \n", len(approx))
            # Først tjek størrelsen, så små elementer udelukkes
            area = cv2.contourArea(contour)
            #print ( "area: \n", area)
            if area > 100:  # Kan justeres afhængig af det forventede kors' størrelse
                # Brug boundingRect for yderligere inspektion
                x, y, w, h = cv2.boundingRect(approx)
                aspectRatio = float(w) / h
                #print ( "aspectRatio: \n", aspectRatio)
                if 0.8 <= aspectRatio <= 1.2:  # Tjek for aspektsforhold tæt på en firkant
                    cv2.drawContours(frame, [approx], -1, (0, 255, 0), 2)
                    cv2.putText(frame, "Obstacle Detected", (x- 30, y - 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    cv2.imshow("Frame with Cross Detection", frame)
                    if not doesBallExistInList(obstacle, x, y): obstacle.append((x, y))
                    # Display coordinates on the frame
                    text = f"X: {x} Y: {y}"
                    cv2.putText(frame, text, (x - 30, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)


    cv2.imshow("Detected circle", frame)
    cv2.imshow("Edges", edges)
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

    
for x,y in egg:
    print(f"Egg Center: ({x}, {y})")     

