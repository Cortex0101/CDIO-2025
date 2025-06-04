
from GetBalls import get_robot_angle
import cv2

cap = cv2.VideoCapture(0)

def test_robot_angle_visual():
    """Test with visual output - press 'q' to quit"""
    global DEBUG
    DEBUG = True  # Enable visual display
    
    print("Testing robot angle detection...")
    print("Position green and yellow objects in camera view")
    print("Press 'q' to quit")
    
    while True:
        angle = get_robot_angle()
        if angle is not None:
            print(f"Current angle: {angle:.1f}°")
        
        # Check for 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()

if __name__ == "__main__":

    test_robot_angle_visual() 

