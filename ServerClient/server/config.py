LARGE_OBJECT_RADIUS = 40  # radius in pixels for large objects like robots when driving
SMALL_OBJECT_RADIUS = 10  # radius in pixels for small objects like balls when collecting

ROBOT_WIDTH = 75 # Guess. Maybe measure?
ROBOT_THICKNESS = 60 # Guess - with claw

OPTIMAL_BALL_LOCATION_DISTANCE = 35

SLOW_EDGE_BALL_MAX_SPEED = 8
SLOW_MAX_SPEED = 20
SLOW_KP = 0.4
SLOW_LOOKAHEAD_DISTANCE = 15  # distance in pixels to look ahead when driving slow
SLOW_MAX_TURN_SLOW = 1

FAST_KP = 0.6
FAST_MAX_SPEED = 30  # max speed in pixels per second when driving fast
FAST_LOOKAHEAD_DISTANCE = 25  # distance in pixels to look ahead when driving fasts
FAST_MAX_TURN_SLOW = 1

TURN_TO_OBJECT_KP = 0.6  # proportional gain for turning to an object
TURN_TO_OBJECT_MAX_SPEED = 30  # max speed in pixels per second when turning to an object

MANUAL_GOAL_CENTER = (549, 211)
USE_MANUAL_GOAL_CENTER = False  # if True, use the manual goal center instead of the one from the course

BALL_STOP_DISTANCE = 30  # distance in pixels to stop before the target object
BALL_STOP_DISTANCE_CROSS = 45  # distance in pixels to stop before the target cross
BALL_STOP_DISTANCE_EDGE_BALL = 20  # distance in pixels to stop before the target edge ball

GOAL_DELIVERY_SPOT_DISTANCE = 75 # prev 75

ROBOT_FACE_OBJECT_ANGLE_THRESHOLD = 3  # degrees, how close the robot needs to be facing the object to consider it "facing" (3 seems to be minimum, otherwise stuck forever)
ROBOT_FACE_OBJECT_ANGLE_THRESHOLD_FRAMES = 10  # how many frames the robot needs to be facing the object to consider it "facing"

REACHED_POINT_DISTANCE = 10  # distance in pixels to consider the robot has reached the point

FORWARD_SPEED_ON_DELIVER = 50  # speed in pixels per second when delivering the ball
OPEN_CLAW_POS_ON_DELIVERY = 30  # position to open the claw when delivering the ball
BALL_SETTLE_TIME_ON_DELIVERY = 1

YOLO_MODEL_15_S = "ball_detect/v15s/weights/best.pt" # Average FPS: 32.40 over 661 frames
YOLO_MODEL_15_M = "ball_detect/v15m/weights/best.pt" # Average FPS: 26.34 over 464 frames 
YOLO_MODEL_15_L = "ball_detect/v15l/weights/best.pt" # Average FPS: 21.44 over 428 frames
YOLO_MODEL_15_X = "ball_detect/v15x/weights/best.pt" # Average FPS: 13.28 over 228 frames (this model was trained for 200 epochs, vs 300 for the others)
YOLO_MODEL_MINIMUM_CONFIDENCE = 0.6  # minimum confidence for object detection to consider it valid

NEAR_WALL_THRESHOLD = 40  # distance in pixels to consider the robot is near a wall
NEAR_CROSS_THRESHOLD = 5  # distance in pixels to consider the robot is near a cross'
SOMEWHAT_NEAR_CROSS_THRESHOLD = 25  # distance in pixels to consider the robot is somewhat near a cross
NEAR_CORNER_THRESHOLD = 40  # distance in pixels to consider the robot is near a corner

OPTIMAL_SPOT_DISTANCE_TO_NON_CROSS = 10
OPTIMAL_SPOT_DISTANCE_TO_CROSS = 40

ROBOT_CLOSED_CLAW_POS = 0  # position to close the claw when collecting the ball
ROBOT_OPEN_CLAW_POS = 90  # position to open the claw when collecting the ball
ROBOT_MIDDLE_CLAW_POS = 45  # position to open the claw when collecting the ball