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


upper_left_corner = []
lower_left_corner = []
upper_right_corner = []
lower_right_corner = []

small_goal = []
big_goal = []


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


# Create trackbars to adjust Blur and Canny values dynamically
cv2.createTrackbar("Canny_thr1", "Canny Trackbars", canny_threshold1, 200, on_trackbar)
cv2.createTrackbar("Canny_thr2", "Canny Trackbars", canny_threshold2, 300, on_trackbar)   
cv2.createTrackbar("gaussian_blur_sigma", "Canny Trackbars", gaussian_blur_sigma, 50, on_trackbar)

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
        if abs (ball[0] - x) < 150 and abs (ball[1] - y) < 150:
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

white_balls_success = 0
obstacle_success = 0;
number_of_frames = 0;


# Afgrænsningsværdier for rød farve i HSV
lower_red_1 = np.array([0, 70, 50])
upper_red_1 = np.array([10, 255, 255])  # Rød nederste område

lower_red_2 = np.array([170, 70, 50])
upper_red_2 = np.array([180, 255, 255])  # Rød øverste område

# Afgrænsningsværdier for hvid farve i HSV
lower_white = np.array([0, 0, 200])
upper_white = np.array([180, 30, 255])


small_goal.clear()
big_goal.clear()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Empty frame!")
        break

    
    # Get the dynamically adjusted HSV values from trackbars
    canny_threshold1 = cv2.getTrackbarPos("Canny_thr1", "Canny Trackbars")
    canny_threshold2 = cv2.getTrackbarPos("Canny_thr2", "Canny Trackbars")  
    gaussian_blur_sigma = cv2.getTrackbarPos("gaussian_blur_sigma", "Canny Trackbars")

    # Convert frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Skab masken for røde farver i to intervaller hjulpet af lys
    #mask1 = cv2.inRange(hsv, lower_red_1, upper_red_1)
    #mask2 = cv2.inRange(hsv, lower_red_2, upper_red_2)
    
    #mask_red = cv2.add(mask1, mask2)

    #mask_white = cv2.inRange(hsv, lower_white, upper_white)

    # Definer hvid maske
    lower_white = np.array([0, 0, 80])
    upper_white = np.array([200, 50, 255])
    mask_white = cv2.inRange(hsv, lower_white, upper_white)


    # Press 'Esc' to exit
    if cv2.waitKey(30) == 27:
        break
 
    #  konvert frame to Gray tones
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    
    # Anvend histogramudligning
    #equalized = cv2.equalizeHist(gray)
   
    # Detecting goals
    # Konverter til gråtoner og slør for at reducere støj
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #blurred = cv2.GaussianBlur(equalized, (gaussian_blur_size_x, gaussian_blur_size_y), gaussian_blur_sigma)

    
    # Anvend GaussianBlur for at reducere støj
    #blurred = cv2.GaussianBlur(mask_red, (5, 5), 0)
    blurred = cv2.GaussianBlur(mask_white, (5, 5), 0)
    # Brug Canny-kantdetektering
    #edges = cv2.Canny(blurred, canny_threshold1, canny_threshold2)
    edges = cv2.Canny(mask_white, 100, 200)
 
   
   
  
   
  

    # Find contours   .CHAIN_APPROX_SIMPLE -> .CHAIN_APPROX_NONE uses more memory
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

     
     
 
     
      

    for contour in contours:

        # Tilnærm polygonens omkreds
        perimeter = cv2.arcLength(contour, True)
        area = cv2.contourArea(contour)
        #print ( "perimeter: \n", perimeter)
        
        
        # Detecting field borders
       

        if perimeter > 3000 :
            #print ( "perimeter: \n", perimeter)
            #print ( "area: \n", area)
            #print ( "contour: \n", contour)
            #print ( "len(contour): \n", len(contour))
         
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

            x, y, w, h = cv2.boundingRect(approx)
            x = int(x + w / 2)
            y = int(y + h / 2) 
            
           # print ( "approx: \n", approx)
            cv2.drawContours(frame, [approx], -1, (0, 255, 0), 2)
            cv2.putText(frame, "Borders Detected", (x- 30, y - 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.imshow("Frame with Cross Detection", frame)
           # if not doesObstacleExistInList(obstacle, x, y): obstacle.append((x, y))
            for i in range (len(approx) ):
                    
                    p1 = tuple(approx[i].flatten())
                    if p1[0] < 200 and p1[1] < 200:       
                     if not doesEndObstExistInList(upper_left_corner, approx[i].flatten()[0],approx[i].flatten()[1]): upper_left_corner.append(approx[i].flatten())
          
                    if p1[0] > 500 and p1[1] < 200:
                        if not doesEndObstExistInList(upper_right_corner, approx[i].flatten()[0],approx[i].flatten()[1]): upper_right_corner.append(approx[i].flatten())

                    if p1[0] > 500 and p1[1] > 500:
                        if not doesEndObstExistInList(lower_right_corner, approx[i].flatten()[0],approx[i].flatten()[1]): lower_right_corner.append(approx[i].flatten())
           
                    if p1[0] < 200 and p1[1] > 500:
                        if not doesEndObstExistInList(lower_left_corner, approx[i].flatten()[0],approx[i].flatten()[1]): lower_left_corner.append(approx[i].flatten())

            if len(upper_left_corner) > 0 and len(upper_right_corner) > 0 and len(lower_left_corner) > 0 and len(lower_right_corner) > 0:  
                        
                    # Display coordinates on the frame
            #  print ( "border_corners: \n", border_corners)
              print ( "upper_left_corner: \n", upper_left_corner)
              print ( "upper_right_corner: \n", upper_right_corner) 
              
              print ( "lower_right_corner: \n", lower_right_corner)
              print ( "lower_left_corner: \n", lower_left_corner)
              
             # small_goal = (lower_right_corner[0] - upper_right_corner[0])/2
              small_goal = upper_left_corner[0][0] , (upper_left_corner[0][1] + lower_left_corner[0][1]) / 2
                # Convert small_goal to a tuple of integers
              small_goal = tuple(map(int, small_goal))
              print("small_goal after conversion:", small_goal)
              text = f"Small goal: {small_goal}" 
       
              
              cv2.putText(frame, text, (small_goal[0] - 30, small_goal[1] - 30),
              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

              #big_goal = (lower_left_corner[0] - upper_left_corner[0])/2
              big_goal = upper_right_corner[0][0], (upper_right_corner[0][1] + lower_right_corner[0][1]) / 2
              big_goal = tuple(map(int, big_goal))
              print("big_goal after conversion:", big_goal)
              text = f"Big goal: {big_goal}"
              cv2.putText(frame, text, (big_goal[0] + 30, big_goal[1] + 30),
              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
      

   
        
        

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
            if 5500 > area > 1000:  # Kan justeres afhængig af det forventede kors' størrelse
            #if 250 > area > 200:
                x, y, w, h = cv2.boundingRect(approx)
                x = int(x + w / 2)
                y = int(y + h / 2)  
                #print ( "area: \n", area)
         
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
                 #   print ( "area: \n", area)
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
                    obstacle_success = obstacle_success + 1

    number_of_frames = number_of_frames + 1
    cv2.imshow("Detected circle", frame)
    cv2.imshow("Edges", edges)
 #   cv2.imshow("Blurred", blurred)
   #  cv2.imshow("Gray", gray)
 #   cv2.imshow("HSV", hsv)
 #   cv2.imshow("Mask", mask)
#    cv2.imshow("Masked HSV", masked_hsv)
upper_left_corner = tuple(map(int, upper_left_corner[0]))
upper_right_corner = tuple(map(int, upper_right_corner[0]))
lower_left_corner = tuple(map(int, lower_left_corner[0]))
lower_right_corner = tuple(map(int, lower_right_corner[0]))


with open("Position2.txt", "w") as f:
        
   f.write(f"Upper_left_corner: {upper_left_corner}\n")
   f.write(f"Upper_right_corner: {upper_right_corner}\n")   
   f.write(f"Lower_left_corner: {lower_left_corner}\n")    
   f.write(f"Lower_right_corner: {lower_right_corner}\n")
   f.write(f"Small_goal: {small_goal}\n")
   f.write(f"Big_goal: {big_goal}\n")
   f.write(f"Obstacle: {obstacle}\n")
    
    



cap.release()
cv2.destroyAllWindows()



print ( "small_goal: \n", small_goal)
print ( "big_goal: \n", big_goal)



for x,y in obstacle:
    print(f"Obstacle Center: ({x}, {y})")
print ("number_of_frames: ", number_of_frames, "number of obstacles detected: ", obstacle_success, "Succes rate: ", 100* obstacle_success/number_of_frames, " pct")   



