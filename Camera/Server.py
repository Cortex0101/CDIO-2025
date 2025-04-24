import socket
import json
import time
import math
from Pathfinding import sort_proximity, calculate_distance, avoid_obstacles
from GetBalls import get_ball_positions, cap

global_robot_size = (15, 15) # bad practice robot size

# used for mocking in get_robot_angle, remove later
global_mock_angle = 0


# ======== PLACEHOLDER CAMERA FUNCTIONS (to be replaced) ========
def get_ball_positions_mock():
    # Replace with actual vision code
    return [(0, 150), (150, 150)]

def get_robot_position():
    # Replace with actual camera tracking code
    return {"x": 0, "y": 0, "theta": 0}

def get_robot_angle():

    #mock shit, remove later
    global global_mock_angle
    # Replace with actual camera tracking code
    return global_mock_angle

def choose_next_ball(balls, current_position):
    # Replace with actual logic to choose the next ball based on proximity
    if not balls:
        return None

    distances = [(calculate_distance(current_position, point), point) for point in balls]
    closest_distance, closest_point = min(distances)

    return closest_point if balls else None

def get_instructions_to_ball(start_position, ball, obstacles=None, obstacle_radius=10):
    # Replace with actual pathfinding logic
    if obstacles == None:
        obstacles = []


    ballX, ballY = ball
    start_position
    startX, startY = start_position
    startpos = (startX, startY)

    current_angle = get_robot_angle

    avoidance_route = avoid_obstacles(startpos, ball, obstacles)

    avoidance_route.insert(0, startpos)

    for i in range(1, len(avoidance_route)):
        
        current_route = avoidance_route[i]

        routeX, routeY = current_route
        startX, startY = startpos

        dx = routeX - startX
        dy = routeY - startY

        target_angle = math.degrees(math.atan2(dy, dx))
        turn_angle = (target_angle - current_angle + 180) % 360 - 180

        distance = calculate_distance(start_position, current_route)

    return [
        {"cmd": "turn", "angle": turn_angle},
        {"cmd": "move", "distance": distance},
    ]

def position_close_enough(actual, expected, threshold=10):
    dx = abs(actual["x"] - expected["x"])
    dy = abs(actual["y"] - expected["y"])
    return dx <= threshold and dy <= threshold

# ===============================================================

HOST = '0.0.0.0'
PORT = 12346

print("[SERVER] Starting camera server...")

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind((HOST, PORT))
server.listen(1)

print(f"[SERVER] Listening on port {PORT}... Waiting for EV3 connection.")
conn, addr = server.accept()
print(f"[SERVER] EV3 robot connected from {addr}")

# === INITIAL SETUP ===
done = False

while not done:
    # Get the ball positions from the camera
    balls = get_ball_positions()
    print(f"[SERVER] Detected balls: {balls}")

    # Choose the next ball to move towards
    next_ball = choose_next_ball(balls, get_robot_position())
    if next_ball:
        print(f"[SERVER] Next ball to move towards: {next_ball}")

        # Get instructions to move towards the chosen ball
        instructions = get_instructions_to_ball(get_robot_position(), next_ball)

        # Send instructions to EV3 robot
        conn.sendall(json.dumps(instructions).encode('utf-8'))

        # wait for status "done" from EV3
        while True:
            data = conn.recv(1024).decode('utf-8')
            if data:
                status = json.loads(data)
                print(f"[SERVER] EV3 status: {status}")
                if status.get("status") == "done":
                    break
            else:
                print("[SERVER] No response from EV3. Retrying...")
                time.sleep(1)

        # Check if the robot is close enough to the ball
        if position_close_enough(get_robot_position(), next_ball):
            print(f"[SERVER] Reached ball at {next_ball}.")
            balls.remove(next_ball) # will no longer be seen by camera

    else:
        done = True
        print("[SERVER] No balls detected. Ending session.")