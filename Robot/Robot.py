from ev3dev2.motor import LargeMotor, OUTPUT_A, OUTPUT_B, MoveTank
from time import sleep
import math

class Robot:
    WHEEL_DIAMETER = 4.2  # cm
    WHEEL_CIRCUMFERENCE = math.pi * WHEEL_DIAMETER  # cm per full wheel rotation
    AXLE_TRACK = 12  # cm (distance between left and right wheels)
    
    def __init__(self):
        self.left_motor = LargeMotor(OUTPUT_A)
        self.right_motor = LargeMotor(OUTPUT_B)
        self.tank_drive = MoveTank(OUTPUT_A, OUTPUT_B)
        
        # Position and orientation tracking
        self.x = 0.0
        self.y = 0.0
        self.angle = 0.0  # Angle in degrees, 0 means facing forward

    def move_forward(self, distance_cm, speed=50):
        rotations = distance_cm / self.WHEEL_CIRCUMFERENCE
        self.tank_drive.on_for_rotations(speed, speed, rotations)
        
        # Update position based on movement direction
        rad_angle = math.radians(self.angle)
        self.x += distance_cm * math.cos(rad_angle)
        self.y += distance_cm * math.sin(rad_angle)
        
    def move_backward(self, distance_cm, speed=50):
        rotations = distance_cm / self.WHEEL_CIRCUMFERENCE
        self.tank_drive.on_for_rotations(-speed, -speed, rotations)
        
        # Update position (moving in the opposite direction)
        rad_angle = math.radians(self.angle)
        self.x -= distance_cm * math.cos(rad_angle)
        self.y -= distance_cm * math.sin(rad_angle)

    def turn_left(self, angle, speed=30):
        turn_distance = (angle / 360) * (math.pi * self.AXLE_TRACK)
        rotations = turn_distance / self.WHEEL_CIRCUMFERENCE
        self.tank_drive.on_for_rotations(-speed, speed, rotations)
        
        # Update orientation
        self.angle = (self.angle + angle) % 360
    
    def turn_right(self, angle, speed=30):
        turn_distance = (angle / 360) * (math.pi * self.AXLE_TRACK)
        rotations = turn_distance / self.WHEEL_CIRCUMFERENCE
        self.tank_drive.on_for_rotations(speed, -speed, rotations)
        
        # Update orientation
        self.angle = (self.angle - angle) % 360

    def get_position(self):
        return self.x, self.y, self.angle
    
    def get_angle(self):
        return self.angle
    
    def test_drive(self):
        print("Starting test drive...")
        
        while True:
            self.move_forward(20, speed=50)
            print(self.get_position())
            sleep(1)
            
            self.turn_right(90, speed=30)
            print(self.get_position())
            sleep(1)
            
            self.move_forward(10, speed=50)
            print(self.get_position())
            sleep(1)
            
            self.turn_left(45, speed=30)
            print(self.get_position())
            sleep(1)
            
            self.move_backward(15, speed=50)
            print(self.get_position())
            sleep(1)

        print("Test drive complete.")