
import math

#ultrasonic = UltrasonicSensor(INPUT_1)

def calculate_distance(point1, point2):
    return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5

def sort_proximity(robot_position, points):
    if not points:
        return []
   
    remaining_points = points.copy()
   
    sorted_points = []

    current_position = robot_position

    while remaining_points:

        # Calculate distances from current position to all remaining points
        distances = [(calculate_distance(current_position, point), point) for point in remaining_points]
       
        closest_distance, closest_point = min(distances)
        sorted_points.append(closest_point)
        remaining_points.remove(closest_point)
        current_position = closest_point
   
    return sorted_points
def line_intersects_obstacle(start, end, obstacle, obstacle_radius):

    dx = obstacle[0] - start[0]
    dy = obstacle[1] - start[1]
        
    line_dx = end[0] - start[0]
    line_dy = end[1] - start[1]

    line_length = calculate_distance(start, end)

    # normalize
    if line_length > 0:
        line_dx /= line_length
        line_dy /= line_length
    else:
        return calculate_distance(start, obstacle) <= obstacle_radius
    
    # closest point on line to object
    t = dx * line_dx + dy * line_dy
    
    # Closest point is beyond line segment
    if t < 0:
        closest_point = start
    elif t > line_length:
        closest_point = end
    else:
        closest_point = (start[0] + t * line_dx, start[1] + t * line_dy)
    
    # Check if closest point is within obstacle
    return calculate_distance(closest_point, obstacle) <= obstacle_radius


def avoid_obstacles(startpoint, endpoint, obstacles, obstacle_radius=10):

    path_clear = True
    for obstacle in obstacles:
        if line_intersects_obstacle(startpoint, endpoint, obstacle, obstacle_radius):
            path_clear = False
            break

    if path_clear:
        return [endpoint]

    # if path is not clear...
    waypoints = []

    for obstacle in obstacles:
        if line_intersects_obstacle(startpoint, endpoint, obstacle, obstacle_radius):

            dx = endpoint[0] - obstacle[0]
            dy = endpoint[1] - obstacle[1]

            # Normalize and scale to create waypoints at a safe distance
            safe_distance = obstacle_radius * 1.5
            norm = calculate_distance((0, 0), (dx, dy))
            if norm > 0:
                dx = dx / norm * safe_distance
                dy = dy / norm * safe_distance
            
            # Create two potential waypoints on either side of the obstacle
            waypoint1 = (obstacle[0] + dy, obstacle[1] - dx)
            waypoint2 = (obstacle[0] - dy, obstacle[1] + dx)
            
            # Choose the waypoint closer to the end point
            if calculate_distance(waypoint1, endpoint) < calculate_distance(waypoint2, endpoint):
                waypoints.append(waypoint1)
            else:
                waypoints.append(waypoint2)

    best_waypoint = None
    best_distance = float('inf')

    for waypoint in waypoints:
        # Check if path to waypoint is clear
        waypoint_path_clear = True
        for obstacle in obstacles:
            if line_intersects_obstacle(startpoint, waypoint, obstacle, obstacle_radius):
                waypoint_path_clear = False
                break
        
        if waypoint_path_clear:
            dist = calculate_distance(waypoint, endpoint)
            if dist < best_distance:
                best_distance = dist
                best_waypoint = waypoint
    
    if best_waypoint:
        # Recursively find path from waypoint to end
        remaining_path = avoid_obstacles(best_waypoint, endpoint, obstacles, obstacle_radius)
        return [best_waypoint] + remaining_path
    
    # failsafe waypoint, greater distance than best_waypoint
    safe_waypoint = (
        (startpoint[0] + endpoint[0]) / 2 + obstacle_radius * 2,
        (startpoint[1] + endpoint[1]) / 2 + obstacle_radius * 2
    )
    return [safe_waypoint, endpoint]

# calibrate real angle based on what the camera angle is
def calibrate_angle(current_angle, camera_input):

    angle_fix = 0.0

    if current_angle != camera_input:

        if current_angle > camera_input:
            angle_fix = current_angle - camera_input

        elif camera_input > current_angle:
            angle_fix = camera_input - current_angle



    return angle_fix

def move_robot(robot, target_points, obstacles=None, wheel_diameter=70, axle_track=165):

    if obstacles is None:
        obstacles = []

    sorted_points = sort_proximity(robot.get_position(), target_points)
    current_x, current_y = robot.get_position()
    current_heading = robot.get_angle()


    for target_x, target_y in sorted_points:
        #dx = target_x - current_x
        #dy = target_y - current_y
        #distance = calculate_distance((current_x, current_y), (target_x, target_y))

        current_position = (current_x, current_y)
        target_position = (target_x, target_y)

        # prints for debugging avoidance code
        print("Positions after get_position call and before object avoidance.")
        print("Current position: " + str(current_position))
        print("Target positon: " + str(target_position))
        
        # Get waypoints for obstacle avoidance
        waypoints = avoid_obstacles(current_position, target_position, obstacles)
        waypoints.insert(0, current_position)



        for i in range(1, len(waypoints)):
            waypoint_x, waypoint_y = waypoints[i]
            
            dx = waypoint_x - current_x
            dy = waypoint_y - current_y
            distance = calculate_distance((current_x, current_y), (waypoint_x, waypoint_y))
            
            target_heading = math.degrees(math.atan2(dy, dx))
            
            turn_angle = (target_heading - current_heading + 180) % 360 - 180

            # prints for debugging avoidance code
            print("Positions after object avoidance.")
            print("Current position: " + str(current_position))
            print("Target positon: " + str(waypoint_x) + ", " + str(waypoint_y))
            print("Turning: " + str(turn_angle))
            
            
            if turn_angle > 0:
                robot.turn_left(turn_angle)
            elif turn_angle < 0:
                robot.turn_right(-turn_angle)
            
            '''if ultrasonic.distance_centimeters < 5:
                robot.move_backward(10)
                break'''
            
            # Move forward
            robot.move_forward(distance)
            

            # Update current position and heading
            current_x, current_y = robot.get_position()
            current_heading = robot.get_angle()

            # prints for debugging avoidance code
            print("Positions after movement but before get() functions.")
            print("Current position: " + str(current_x) + ", " + str(current_y))
            print("Target positon: " + str(waypoint_x) + ", " + str(waypoint_y))
            print("Heading after moving: " + str(current_heading))

    print("Navigation completed")
