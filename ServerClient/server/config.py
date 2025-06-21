LARGE_OBJECT_RADIUS = 40  # radius in pixels for large objects like robots when driving
SMALL_OBJECT_RADIUS = 10  # radius in pixels for small objects like balls when collecting

SLOW_EDGE_BALL_MAX_SPEED = 8
SLOW_MAX_SPEED = 15
SLOW_KP = 0.3

MANUAL_GOAL_CENTER = (370, 465)

BALL_STOP_DISTANCE = 20  # distance in pixels to stop before the target object
BALL_STOP_DISTANCE_EDGE_BALL = 20  # distance in pixels to stop before the target edge ball

GOAL_DELIVERY_SPOT_DISTANCE = 90 # prev 75

ROBOT_FACE_OBJECT_ANGLE_THRESHOLD = 3  # degrees, how close the robot needs to be facing the object to consider it "facing" (3 seems to be minimum, otherwise stuck forever)
ROBOT_FACE_OBJECT_ANGLE_THRESHOLD_FRAMES = 5  # how many frames the robot needs to be facing the object to consider it "facing"

REACHED_POINT_DISTANCE = 10  # distance in pixels to consider the robot has reached the point

FORWARD_SPEED_ON_DELIVER = 50  # speed in pixels per second when delivering the ball
OPEN_CLAW_POS_ON_DELIVERY = 55  # position to open the claw when delivering the ball