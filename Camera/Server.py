# Import the necessary modules for testing
import socket
import json
import time
import math
from Pathfinding import sort_proximity, calculate_distance, avoid_obstacles
if __name__ == "__main__":
    from GetBalls import get_objects, cap, get_robot_position, get_robot_angle

global_robot_size = (15, 15) # bad practice robot size

# used for mocking in get_robot_angle, remove later
global_mock_angle = 0

def choose_next_ball(white_balls, orange_balls, current_position):
    # Replace with actual logic to choose the next ball based on proximity
    if not white_balls and not orange_balls:
        return None

    # TODO Include orange balls in the logic
    distances = [(calculate_distance(current_position, point), point) for point in white_balls]
    closest_distance, closest_point = min(distances)

    return closest_point if white_balls else None

def get_instructions_to_ball(start_position, ball, obstacles=None, obstacle_radius=10):
    # Replace with actual pathfinding logic
    if obstacles == None:
        obstacles = []


    ballX, ballY = ball
    start_position
    startX, startY = start_position
    startpos = (startX, startY)

    current_angle = get_robot_angle()

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
    dx = abs(actual[0] - expected[0])
    dy = abs(actual[1] - expected[1])
    return dx <= threshold and dy <= threshold

# Server code wrapped inside the `if __name__ == "__main__":` block
if __name__ == "__main__":
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
        objects = get_objects()
        print(f"[SERVER] Detected balls: {objects["white_balls"] + objects["orange_balls"]}")

        # Choose the next ball to move towards
        rbt_pos = get_robot_position()
        next_ball = choose_next_ball(objects["white_balls"], objects["orange_balls"], rbt_pos)
        if next_ball:
            print(f"[SERVER] Next ball to move towards: {next_ball}")

            # Get instructions to move towards the chosen ball
            instructions = get_instructions_to_ball(get_robot_position(), next_ball)

            # Send instructions to EV3 robot
            cmds = json.dumps(instructions).encode('utf-8')
            
            #{"cmd": "turn", "angle": turn_angle},
            #{"cmd": "move", "distance": distance},
            
            # send the instruction one at a time
            for cmd in instructions:
                print(f"[SERVER] Sending command: {cmd}")
                json_cmd = json.dumps(cmd).encode('utf-8')
                conn.sendall(json_cmd)

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
            print(f"[SERVER] Checking if robot is close enough to the ball...")
            print(f"Robot position: {get_robot_position()}, Next ball position: {next_ball}, Robot angle: {get_robot_angle()}")
            if position_close_enough(get_robot_position(), next_ball):
                print(f"[SERVER] Reached ball at {next_ball}.")

        else:
            done = True
            print("[SERVER] No balls detected. Ending session.")