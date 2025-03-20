import cv2 as cv
import numpy as np
import math

def detect_direction(cap, color1_hsv, color2_hsv):
    if not cap.isOpened(): 
        print("Camera error")
        return None
    
    angle = None
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        # makes masks[color]
        masks = {color: cv.inRange(hsv, *hsv_range) for color, hsv_range in zip(('color1', 'color2'), (color1_hsv, color2_hsv))}
        centers = {}
        
        for color, mask in masks.items():
            contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            if contours:
                largest = max(contours, key=cv.contourArea)
                # area size to catch, currently: 100 pixels
                if cv.contourArea(largest) > 100:
                    M = cv.moments(largest)
                    centers[color] = (int(M['m10']/M['m00']), int(M['m01']/M['m00']))
        
        result = np.zeros_like(frame)
        if 'color1' in centers and 'color2' in centers:
            cv.line(result, centers['color2'], centers['color1'], (255, 255, 255), 2)
            dx, dy = np.subtract(centers['color1'], centers['color2'])
            # change dy, dx if e.g. 90 degrees is now up instead of right etc.
            angle = (math.degrees(math.atan2(dy, dx)) + 360) % 360
            cv.putText(result, f"{angle:.1f}°", centers['color1'], cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            print(f"Direction: {angle:.1f}°")
        
        cv.imshow('Direction', result)
        if cv.waitKey(1) & 0xFF == ord('q'): break
    
    cap.release()
    cv.destroyAllWindows()
    return angle

def main():
    cap = cv.VideoCapture(0)
    # green range
    green_hsv = (np.array([70, 100, 50]), np.array([95, 255, 200]))
    # yellow range
    yellow_hsv = (np.array([20, 100, 100]), np.array([35, 255, 255]))
    angle = detect_direction(cap, green_hsv, yellow_hsv)
    print(f"Final detected direction: {angle}°")

if __name__ == "__main__":
    main()