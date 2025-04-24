import socket
import json
import time
from Pathfinding import sort_proximity, calculate_distance

# ======== PLACEHOLDER CAMERA FUNCTIONS (to be replaced) ========
def get_ball_positions():
    # Replace with actual vision code
    return [(0, 150), (150, 150)]

def get_robot_position():
    # Replace with actual camera tracking code
    return {"x": 0, "y": 0, "theta": 0}

def get_robot_angle():
    # Replace with actual camera tracking code
    return 0

def choose_next_ball(balls, current_position):
    # Replace with actual logic to choose the next ball based on proximity

    distances = [(calculate_distance(current_position, point), point) for point in balls]
    closest_distance, closest_point = min(distances)

    return closest_point if balls else None

def get_instructions_to_ball(start_position, ball):
    # Replace with actual pathfinding logic
    distance = calculate_distance(start_position, ball)


    return [
        {"cmd": "turn", "angle": 90},
        {"cmd": "move", "distance": 150},
    ]

def position_close_enough(actual, expected, threshold=10):
    dx = abs(actual["x"] - expected["x"])
    dy = abs(actual["y"] - expected["y"])
    return dx <= threshold and dy <= threshold

# ===============================================================

HOST = '0.0.0.0'
PORT = 12345

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