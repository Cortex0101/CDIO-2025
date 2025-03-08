


def sort_proximity(robot_position, points):
    if not points:
        return []
    
    # Make a copy of points to avoid modifying the original
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

def calculate_distance(point1, point2):
    return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5

if __name__ == "__main__":

    # example
    robot_pos = (0, 0)
    
    target_points = [(3, 4), (7, 7), (5, 1), (2, 3)]
    
    sorted_points = sort_proximity(robot_pos, target_points)
    
    print(f"Robot starting position: {robot_pos}")
    print(f"Original points: {target_points}")
    print(f"Sorted points: {sorted_points}")

    print("\nPath:")
    current = robot_pos
    print(f"Robot at {current}")
    for point in sorted_points:
        print(f"Moving to {point}, distance: {calculate_distance(current, point):.2f}")
        current = point