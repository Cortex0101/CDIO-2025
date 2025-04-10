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

    print("Punkter: ", a, b, c, "\n")
    
    ab = a - b
    cb = c - b
    #cb = b - c

    print("Vektorer: ", ab, cb, "\n")
    # Beregn vinklen mellem vektorerne ab og cb
    
    cosine_angle = np.dot(ab, cb) / (np.linalg.norm(ab) * np.linalg.norm(cb))
    if cosine_angle < -1 :
        cosine_angle = -1
    if cosine_angle > 1 :
        cosine_angle = 1
    #cosine_angle = ab/cb
    angle = np.arccos(cosine_angle)
    print("Cosine angle: ", cosine_angle, "\n")
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
    
def corner_detection(angels):
    inner_corner = 0
    outer_corner = 0
    for i in range(len(angels)):
        if angels[i] > 130 and angels[i] < 140 or angels[i] < -130 and angels[i] > -140:
            inner_corner += 1
        elif angels[i] > 80 and angels[i] < 100 or angels[i] < -80 and angels[i] > -100:
            outer_corner += 1
    return inner_corner, outer_corner


def find_cross_points(contour):
    min_x = min_y = float('inf')
    max_x = max_y = float('-inf')
    
    for point in contour:
        x, y = point[0]
        if x < min_x:
            min_x = x
        if y < min_y:
            min_y = y
        if x > max_x:
            max_x = x
        if y > max_y:
            max_y = y
    
    return (np.int32 (min_x), np.int32 (min_y)), (np.int32 (max_x), np.int32 (min_y)), (np.int32 (min_x), np.int32 (max_y)), (np.int32 (max_x), np.int32 (max_y))

# Funktion til at udglatte ved hjælp af glidende gennemsnit
def moving_average(data, window_size):
    # Beregn glidende gennemsnit
    cumsum = np.cumsum(data)
    cumsum[window_size:] = cumsum[window_size:] - cumsum[:-window_size]
    return cumsum[window_size - 1:] / window_size

#def calculate_angle(a, b, c):
#    # Beregner vinklen ABC (i grader)
#    ab = a - b
#    cb = c - b
#    cosine_angle = np.dot(ab, cb) / (np.linalg.norm(ab) * np.linalg.norm(cb))
#    print("Cosine angle: ", cosine_angle, "\n")
#    # Sørg for at cosine_angle er inden for [-1, 1] intervallet
#    angle = np.arccos(cosine_angle)
#    return np.degrees(angle)

#def find_cross_angles(contour):
#    angles_and_points = []
#    contour = contour.reshape(-1, 2)
    
    # Iterér over konturens punkter
#    for i in range(len(contour)):
#        p1 = contour[i - 1]  # Point før
#        p2 = contour[i]      # Nuværende point
#        p3 = contour[(i + 1) % len(contour)]  # Punkt efter
#        
#        angle = calculate_angle(p1, p2, p3)
#        
#        # Analyser væsentlige vinkler
#        if 40 < angle < 50:  # Justér, hvis anledning kræver det
#            angles_and_points.append((angle, tuple(p2)))
    
#    return angles_and_points


def find_cross_corners(contour):
    contour = contour.reshape(-1, 2)
    
    # Find centroiden af konturen
    M = cv2.moments(contour)
    if M["m00"] != 0:  # For at undgå division med nul
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        return []

    # Find hjørne punkter ved at analysere afstanden til centroiden
    distances = np.linalg.norm(contour - np.array([cX, cY]), axis=1)
    far_points = contour[np.argsort(distances)[-4:]]  # Fjerneste punkter fra centrum

    # Sortér punkterne (aktuel rækkefølge kan variere)
    far_points = sorted(far_points, key=lambda x: (np.arctan2(x[1] - cY, x[0] - cX)))

    return far_points

def rotate_image(image, angle, center=None):
    # Find center for rotation
    if center is None:
        center = (image.shape[1] // 2, image.shape[0] // 2)

    # Rotate the image
    M = cv.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv.warpAffine(image, M, (image.shape[1], image.shape[0]))

    return rotated

def find_angle(points):
    # Beregn vinklen til at dreje korset til at flugte med akserne
    # Antager, at de to yderste punkter i korset er (x1, y1) og (x2, y2)
    (x1, y1), (x2, y2) = points

    if x2 - x1 == 0:  # Undgå division med nul
        return 90  # lodret linje
    angle = np.arctan2(y2 - y1, x2 - x1) * (180.0 / np.pi)
    return angle





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
   # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
   


    # Press 'Esc' to exit
    if cv2.waitKey(30) == 27:
        break
 
    #  konvert frame to Gray tones and HSV
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   
    # Detecting goals
    # Konverter til gråtoner og slør for at reducere støj
 #   gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 #   blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Brug Canny-kantdetektering
 #   edges = cv2.Canny(blurred, 50, 150)

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
   
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Reduce noise
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    #blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Do Canny edgedetektion
    #edges = cv2.Canny(blurred, 30, 150)
    edges = cv2.Canny(blurred, 50, 100)

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
          if len(approx) > 5 and len(approx) < 8 :  # approximation of a circle
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
                
                if mean_val[1] < 140:
                #if mean_val[1] < 120:
                #if  mean_val[0] > 210  and mean_val[1] < 150 :
                    color = "White"
                    #print ( "White mean_val: \n", mean_val)
                elif (5 < mean_val[0] < 40) and (150 < mean_val[1] < 255):
                #elif (10 < mean_val[0] < 50) and (50 < mean_val[1] < 210):
                    color = "Orange"
                    #print ( "mean_val: \n", mean_val)
                else:
                   # print ( "undefined mean_val: \n", mean_val)
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
        #approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
         #Kontroller for rektangulært formet objekt
        n = len(approx)
        #n = len(contour)
        #print ( "approx: \n", approx)
        #print ( "len(contour): \n", n)
        if 8 <= n  < 4000 :
       # if 350 <= n  < 700 :  # Antag at det kræver mange linjer i det detaljerede kors
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
                top_left, top_right, bottom_left, bottom_right = find_cross_points(approx)
                print(f"Top Left: {top_left}, Top Right: {top_right}, Bottom Left: {bottom_left}, Bottom Right: {bottom_right}")
                cross_detected = True
                angles = []

                #print ( "len approx: \n", len (approx))
                
                with open("contour.txt", "w") as f:
                    for i in range(n):
                        f.write(f"{approx[i][0]}\n")
               
                with open("p.txt", "w") as f:
                    f.write(f"Contour points: {n}\n")
                    for i in range(n - 4):
                        p1 = tuple(approx[i][0])
                        p2 = tuple(approx[(i+1) % n][0])
                        p3 = tuple(approx[(i+2) % n][0])
                        f.write(f"{p1}   {p2}   {p3}   \n")
               
                
               # Antager de to punkter ligger i modsatte ender af korset
               # angle = find_angle(contour)
                
               # Rotér billedet
                #rotated_image = rotate_image(contour, angle)  # hvor original_image er dit billede
          
               # Find hjørne punkter ved at analysere afstanden til centroiden
                #distances = np.linalg.norm(approx - np.array([x, y]), axis=1)
               # far_points = contour[np.argsort(distances)[-8:]]  # Fjerneste punkter fra centrum
                end_obst.clear()
                for i in range (len(approx) ):
                    p1 = tuple(approx[i])
                    #p2 = tuple ([x,y])
                    distp = np.linalg.norm(p1 - np.array([x, y]))
                    print ( "distp: \n", distp)
                    if distp > 15:
                       if not doesEndObstExistInList(end_obst, approx[i].flatten()[0],approx[i].flatten()[1]): end_obst.append(approx[i].flatten())

                print ( "end_obst: \n", end_obst)
                
               # Sortér punkterne (aktuel rækkefølge kan variere)
                #far_points = sorted(far_points, key=lambda z: (np.arctan2(z[1] - y, z[0] - x)))

                #print ( "far_points: \n", far_points)
    
                p2 = tuple ([x,y])
                
                for i in range (len(end_obst) - 1):
                   
                    print ( "i: \n", i)
                                        
                    angle = ideal_angle_between_points (angle_between_points(end_obst[i], p2, end_obst[i + 1]))
                
                    
                    angles.append(angle)
                    
                    smooth_angles = moving_average (angles, 5)
                    inner_corner, outer_corner = corner_detection(angles)
                    
                if len(end_obst) > 2:
                  angle = ideal_angle_between_points (angle_between_points(end_obst[i + 1], p2, end_obst[0]))
                  angles.append(angle)
                else:
                  cross_detected = False
                ninety_degree = 4
                with open("angles.txt", "w") as f:
                    f.write(f"Angles: {n}\n")
                    for angle in angles:
                        if angle != 90:
                            ninety_degree -= 1
                        f.write(f"{angle}\n")  
                 
                if ninety_degree < 4:
                    cross_detected = False
               
                if cross_detected :
                    print ( "Cross detected. \n")
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

