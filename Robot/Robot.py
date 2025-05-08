from ev3dev2.motor import LargeMotor, MediumMotor, OUTPUT_A, OUTPUT_B, OUTPUT_C, OUTPUT_D, MoveTank

from ev3dev2.sound import Sound
from time import sleep
import math

class Robot:
    WHEEL_DIAMETER = 5.5 # cm
    WHEEL_CIRCUMFERENCE = math.pi * WHEEL_DIAMETER  # cm per full wheel rotation
    AXLE_TRACK = 16.5  # cm (distance between left and right wheels)

    CLAW_OPEN_POS = 0
    CLAW_CLOSED_POS = 90
    
    def __init__(self):
        self.left_motor = LargeMotor(OUTPUT_C)
        self.right_motor = LargeMotor(OUTPUT_D)
        self.claw_motor = MediumMotor(OUTPUT_B)
        self.tank_drive = MoveTank(OUTPUT_C, OUTPUT_D)

    def move_forward(self, distance_cm, speed=50):
        rotations = distance_cm / self.WHEEL_CIRCUMFERENCE
        self.tank_drive.on_for_rotations(speed, speed, rotations)
        
    def move_backward(self, distance_cm, speed=50):
        rotations = distance_cm / self.WHEEL_CIRCUMFERENCE
        self.tank_drive.on_for_rotations(-speed, -speed, rotations)

    def turn_left(self, angle, speed=30):
        turn_distance = (angle / 360) * (math.pi * self.AXLE_TRACK)
        rotations = turn_distance / self.WHEEL_CIRCUMFERENCE
        self.tank_drive.on_for_rotations(-speed, speed, rotations)

    def turn_right(self, angle, speed=30):
        turn_distance = (angle / 360) * (math.pi * self.AXLE_TRACK)
        rotations = turn_distance / self.WHEEL_CIRCUMFERENCE
        self.tank_drive.on_for_rotations(speed, -speed, rotations)

    def open_claw(self):
        self.claw_motor.on_to_position(speed=20, position_sp=self.CLAW_OPEN_POS)

    def close_claw(self):
        self.claw_motor.on_to_position(speed=20, position_sp=self.CLAW_CLOSED_POS)

    