import cv2
import numpy as np

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

def get_ball_positions():
    ret, frame = cap.read()

    if not ret:
        print("Error: Unable to read frame from camera.")
        return None

  
    # Parameters
    gaussian_blur_size = (11, 11)
    gaussian_blur_sigma = 0
    canny_threshold1 = 39
    canny_threshold2 = 100

    
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

    # Convert to grayscale and blur
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, gaussian_blur_size, gaussian_blur_sigma)

    # Edge detection
    edges = cv2.Canny(blurred, canny_threshold1, canny_threshold2)
    

    # Convert to HSV for color classification
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    def circularity(area, perimeter):
        return 4 * np.pi * (area / (perimeter * perimeter)) if perimeter != 0 else 0

    def is_close(ballList, x, y, tolerance):
        for bx, by in ballList:
            if abs(bx - x) < tolerance and abs(by - y) < tolerance:
                return True
        return False
    
    def doesEndObstExistInList(obstList, x, y):
     #  print ( "obstList: \n", obstList)
     #  print ( "x: ", x, "y: \n", y)   
      for ball in obstList:
        if abs (ball[0] - x) < 15 and abs (ball[1] - y) < 15:
            return True
      return False
    
    def ideal_angle_between_points(p):
    
     if  80 < p < 100: 
        return 90
     else:   
        return p


    
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
   
   
   
    def analyze_contours(contours):
  
        white_balls = []
        orange_balls = []
        obstacle = []
        egg = []
  
        for contour in contours:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)

            if perimeter < 10 or perimeter > 500:
                continue

            approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
            ((x, y), radius) = cv2.minEnclosingCircle(approx)
            center = (int(x), int(y))
            circ = circularity(area, perimeter)

            # Detect Balls
            if 0.8 < circ < 1.0 and 6 < radius < 12 and len(approx) > 4 and len(approx) < 8:
                mask = np.zeros_like(hsv)
                cv2.circle(mask, center, int(radius), (255, 255, 255), -1)
                mean_val = cv2.mean(hsv, mask[:,:,0])

                if 70 < mean_val[0] < 140 and 5 < mean_val[1] < 100:
                    if not is_close(white_balls, int(x), int(y), 15):
                        white_balls.append((int(x), int(y)))
                elif 10 < mean_val[0] < 50 and 50 < mean_val[1] < 150:
                    if not is_close(orange_balls, int(x), int(y), 15):
                        orange_balls.append((int(x), int(y)))
            
            # Detect Egg (larger, slightly elongated circle)
            if 0.8 < circ < 0.9 and 15 < radius < 25:
                if not is_close(egg, int(x), int(y), 15):
                    egg.append((int(x), int(y)))

            # Detect Obstacle-like shape (cross with angles)
  
  
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
                x, y, w, h = cv2.boundingRect(approx)
                x = int(x + w / 2)
                y = int(y + h / 2)  
         
         
                cross_detected = True
                angles = []

                #print ( "len approx: \n", len (approx))
                
                
                end_obst = []
    
                
               
                end_obst.clear()
                for i in range (len(approx) ):
                    p1 = tuple(approx[i])
                    #p2 = tuple ([x,y])
                    distp = np.linalg.norm(p1 - np.array([x, y]))
                 #   print ( "distp: \n", distp)
                    if distp > 15:
                       if not doesEndObstExistInList(end_obst, approx[i].flatten()[0],approx[i].flatten()[1]): end_obst.append(approx[i].flatten())

                print ( "end_obst: \n", end_obst)
                
               
    
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
             
                 
                if ninety_degree < 4:
                    cross_detected = False
               

                if cross_detected == True and not is_close(obstacle, x, y, 50):
                    obstacle.append((x, y))
        
        return white_balls, orange_balls, obstacle, egg
 
    white_balls, orange_balls, obstacle, egg = analyze_contours(contours)
    return {
        "white_balls": white_balls,
        "orange_balls": orange_balls,
        "obstacles": obstacle,
        "eggs": egg
    }

