LARGE_OBJECT_RADIUS = 40  # radius in pixels for large objects like robots when driving
SMALL_OBJECT_RADIUS = 10  # radius in pixels for small objects like balls when collecting

SLOW_EDGE_BALL_MAX_SPEED = 8
SLOW_MAX_SPEED = 15
SLOW_KP = 0.3

MANUAL_GOAL_CENTER = (370, 465)

BALL_STOP_DISTANCE = 20  # distance in pixels to stop before the target object
BALL_STOP_DISTANCE_EDGE_BALL = 20  # distance in pixels to stop before the target edge ball

GOAL_DELIVERY_SPOT_DISTANCE = 70 # prev 75

ROBOT_FACE_OBJECT_ANGLE_THRESHOLD = 3  # degrees, how close the robot needs to be facing the object to consider it "facing" (3 seems to be minimum, otherwise stuck forever)
ROBOT_FACE_OBJECT_ANGLE_THRESHOLD_FRAMES = 5  # how many frames the robot needs to be facing the object to consider it "facing"

REACHED_POINT_DISTANCE = 10  # distance in pixels to consider the robot has reached the point

FORWARD_SPEED_ON_DELIVER = 50  # speed in pixels per second when delivering the ball
OPEN_CLAW_POS_ON_DELIVERY = 55  # position to open the claw when delivering the ball
BALL_SETTLE_TIME_ON_DELIVERY = 2

YOLO_MODEL_15_S = "ball_detect/v15s/weights/best.pt" # avg
YOLO_MODEL_15_M = "ball_detect/v15m/weights/best.pt" # avg 
YOLO_MODEL_15_L = "ball_detect/v15l/weights/best.pt" # avg 
YOLO_MODEL_15_X = "ball_detect/v15x/weights/best.pt" # avg (this model was trained for 200 epochs, vs 300 for the others)
YOLO_MODEL_MINIMUM_CONFIDENCE = 0.5  # minimum confidence for object detection to consider it valid

NEAR_WALL_THRESHOLD = 40  # distance in pixels to consider the robot is near a wall
NEAR_CROSS_THRESHOLD = 5  # distance in pixels to consider the robot is near a cross
NEAR_CORNER_THRESHOLD = 40  # distance in pixels to consider the robot is near a corner