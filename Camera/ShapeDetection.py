import cv2
import numpy as np



# Lists to store detected balls
white_balls = []
orange_balls = []

# Obstacle detection
obstacle_x, obstacle_y = 0, 0
obstacle = []

end_obst = []

egg = []


# Callback function for trackbars (does nothing but needed for trackbars)
def on_trackbar(val):
    pass

# Create a window for trackbars
cv2.namedWindow("Canny Trackbars")
Canny_threshold1 = 50
Canny_threshold2 = 150



Gaussian_blur_size_x = 11
Gaussian_blur_size_y = 11
Gaussian_blur_sigma = 0

# Create trackbars to adjust Blur and Canny values dynamically
cv2.createTrackbar("Canny_thr1", "Canny Trackbars", Canny_threshold1, 200, on_trackbar)
cv2.createTrackbar("Canny_thr2", "Canny Trackbars", Canny_threshold2, 300, on_trackbar)
cv2.createTrackbar("Gaussian_blur_size_x", "Canny Trackbars", Gaussian_blur_size_x, 50, on_trackbar)
cv2.createTrackbar("Gaussian_blur_size_y", "Canny Trackbars", Gaussian_blur_size_y, 50, on_trackbar)    
cv2.createTrackbar("Gaussian_blur_sigma", "Canny Trackbars", Gaussian_blur_sigma, 50, on_trackbar)

def doesBallExistInList(ballList, x, y):
    for ball in ballList:
        if abs (ball[0] - x) < 15 and abs (ball[1] - y) < 15:
            return True
    return False


def doesObstacleExistInList(obstList, x, y):
    for ball in obstList:
        if abs (ball[0] - x) < 50 and abs (ball[1] - y) < 50:
            return True
    return False



def doesEndObstExistInList(obstList, x, y):
  #  print ( "obstList: \n", obstList)
  #  print ( "x: ", x, "y: \n", y)   
    for ball in obstList:
        if abs (ball[0] - x) < 15 and abs (ball[1] - y) < 15:
            return True
    return False


# Callback function for trackbars (does nothing but needed for trackbars)
def on_trackbar(val):
    pass

def angle_between_points(p1, p2, p3):
    # Finder vinklen mellem tre punkter: p1, p2 (vertex), p3
    a = np.array(p1)
    b = np.array(p2)
    c = np.array(p3)

    #print("Punkter: ", a, b, c, "\n")
    
    ab = a - b
    cb = c - b
    

    #print("Vektorer: ", ab, cb, "\n")
    # Beregn vinklen mellem vektorerne ab og cb
    
    cosine_angle = np.dot(ab, cb) / (np.linalg.norm(ab) * np.linalg.norm(cb))
    if cosine_angle < -1 :
        cosine_angle = -1
    if cosine_angle > 1 :
        cosine_angle = 1
    
    angle = np.arccos(cosine_angle)
    #print("Cosine angle: ", cosine_angle, "\n")
    return np.degrees(angle)

def ideal_angle_between_points(p):
    if -3 < p < 3:
        return 0
    elif    41 < p < 48:
        return 45
    elif  80 < p < 100: 
        return 90
    elif 133 < p < 137: 
        return 135
    elif 178 < p < 182: 
        return 180
    elif 223 < p < 227: 
        return 225
    elif 268 < p < 272: 
        return 270
    elif 313 < p < 317: 
        return 315
    elif 358 < p < 362: 
        return 360
    else:   
        return p
 





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

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Empty frame!")
        break

    

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
    blurred = cv2.GaussianBlur(gray, (Gaussian_blur_size_x, Gaussian_blur_size_y), Gaussian_blur_sigma)

    # Brug Canny-kantdetektering
    edges = cv2.Canny(blurred, Canny_threshold1, Canny_threshold2)

    # Find konturer i kanten
 #   contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

 #   for contour in contours:
        # Approximér hver kontur
 #       perimeter = cv2.arcLength(contour, True)
 #       approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

        # Kontroller for rektangulært formet objekt
  #      if len(approx) >= 12:  # Antag at det kræver mange linjer i det detaljerede kors
            # Først tjek størrelsen, så små elementer udelukkes
  #          area = cv2.contourArea(contour)
 #           if area > 100:  # Kan justeres afhængig af det forventede kors' størrelse
                # Brug boundingRect for yderligere inspektion
 #               x, y, w, h = cv2.boundingRect(approx)
 #               aspectRatio = float(w) / h
 #               if 0.8 <= aspectRatio <= 1.2:  # Tjek for aspektsforhold tæt på en firkant
 #                   cv2.drawContours(frame, [approx], -1, (0, 255, 0), 2)
 #                   cv2.putText(frame, "Cross Detected", (x, y - 10),
 #                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
   
 #       cv2.imshow("Goal Detection", frame)
   
   
    # Detecting balls, obstacles and eggs
   
  

    # Find contours   .CHAIN_APPROX_SIMPLE -> .CHAIN_APPROX_NONE uses more memory
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for contour in contours:
        # Tilnærm polygonens omkreds
        perimeter = cv2.arcLength(contour, True)
        area = cv2.contourArea(contour)
        #print ( "perimeter: \n", perimeter)
        if perimeter > 10 and perimeter < 500  :
          #print ( "perimeter: \n", perimeter)
          #For detecting balls
          approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
          #print ( "approx: \n", approx)
          #print ( "Len approx: \n", len(approx))
        
          # Check if the kocontour is circle
          if len(approx) > 4 and len(approx) < 8 :  # approximation of a circle
            ((x, y), radius) = cv2.minEnclosingCircle(approx)
            center = (int(x), int(y))
            # print ( "center: \n", center)
            #print ( "radius: \n", radius)
            
            # check if the circle is really a ball
            
              # Beregn cirkularitet
            if perimeter == 0: continue  # Undgå division med nul
            circularity = 4 * np.pi * (area / (perimeter * perimeter))

            # Hvis cirkularitet er tæt på 1, betragtes det som en cirkel
            if 0.8 < circularity < 1.0:  # Juster grænser til præference og præcision
            # Omkreds og enkod cirkel
               ((x, y), radius) = cv2.minEnclosingCircle(contour)
              # avoid small detections
               if radius > 6 and radius < 12:
                mask = np.zeros_like(hsv)
                #mask = np.zeros_like(gray)
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
                #if mean_val[1] < 120:
                #if  mean_val[0] > 210  and mean_val[1] < 150 :
                if  70 < mean_val[0] < 140  and 50 < mean_val[1] < 100 :
                    color = "White"
                 #   print ( "White mean_val: \n", mean_val)
                #elif (5 < mean_val[0] < 40) and (150 < mean_val[1] < 255):
                elif (10 < mean_val[0] < 50) and (50 < mean_val[1] < 150):
                    color = "Orange"
                 #   print ( "Orange mean_val: \n", mean_val)
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
            if radius > 20 and radius < 25 and 0.8 < circularity < 0.9:
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
        
        n = len(approx)
        #n = len(contour)
        #print ( "approx: \n", approx)
        #print ( "len(contour): \n", n)
        if 8 <= n  < 15 :
            #print ( "len(approx): \n", len(approx))
            #print ( "contour: \n", contour)
            # Først tjek størrelsen, så små elementer udelukkes
            area = cv2.contourArea(contour)
            #print ( "area: \n", area)
            if 5500 > area > 70:  # Kan justeres afhængig af det forventede kors' størrelse
            #if 250 > area > 200:
                x, y, w, h = cv2.boundingRect(approx)
                x = int(x + w / 2)
                y = int(y + h / 2)  
         
         
                cross_detected = True
                angles = []

                #print ( "len approx: \n", len (approx))
                
                    
                
               
                end_obst.clear()
                for i in range (len(approx) ):
                    p1 = tuple(approx[i])
                    #p2 = tuple ([x,y])
                    distp = np.linalg.norm(p1 - np.array([x, y]))
                 #   print ( "distp: \n", distp)
                    if distp > 15:
                       if not doesEndObstExistInList(end_obst, approx[i].flatten()[0],approx[i].flatten()[1]): end_obst.append(approx[i].flatten())

               # print ( "end_obst: \n", end_obst)
                
               
    
                p2 = tuple ([x,y])
                ninety_degree = 4

                for i in range (len(end_obst) - 1):
                   
                  #  print ( "i: \n", i)
                                        
                    angle = ideal_angle_between_points (angle_between_points(end_obst[i], p2, end_obst[i + 1]))
                          
                    if angle != 90:
                      ninety_degree -= 1
                    
                    angles.append(angle)
                    
                
                    
                if len(end_obst) > 2:
                  angle = ideal_angle_between_points (angle_between_points(end_obst[i + 1], p2, end_obst[0]))
                  angles.append(angle)
                else:
                  cross_detected = False
                
             #   with open("angles.txt", "w") as f:
             #       f.write(f"Angles: {n}\n")
             #       for angle in angles:
                        
                            
             #           f.write(f"{angle}\n")  
                 
                if ninety_degree < 4:
                    cross_detected = False
               
                if cross_detected :
              #      print ( "Cross detected. \n")
                    cv2.drawContours(frame, [approx], -1, (0, 255, 0), 2)
                    cv2.putText(frame, "Obstacle Detected", (x- 30, y - 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    cv2.imshow("Frame with Cross Detection", frame)
                    if not doesObstacleExistInList(obstacle, x, y): obstacle.append((x, y))
                    # Display coordinates on the frame
                    text = f"X: {x} Y: {y}"
                    cv2.putText(frame, text, (x - 30, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)


    cv2.imshow("Detected circle", frame)
    cv2.imshow("Edges", edges)
 #   cv2.imshow("Blurred", blurred)
   #  cv2.imshow("Gray", gray)
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

