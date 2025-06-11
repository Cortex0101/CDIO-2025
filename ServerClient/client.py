#!/usr/bin/env python3

from ev3dev2.motor import LargeMotor, MediumMotor, OUTPUT_A, OUTPUT_B, OUTPUT_C, OUTPUT_D, MoveTank
from ev3dev2.sound import Sound
from time import sleep
import math
import socket
import json

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

        self.claw_motor.position = self.CLAW_OPEN_POS  # Initialize claw position to open

    def move_forward(self, left_speed, right_speed):
        self.tank_drive.on(left_speed, right_speed)

    # just use negative values for backwards
    def move_in_cm(self, distance_cm):
        self.tank_drive.on_for_distance(left_speed=distance_cm, right_speed=distance_cm, distance=distance_cm)

    def perform_jiggle(self, number_of_jiggles=2, jiggle_degrees=4):
        """
        Jiggles the robot left and right by first turning left wheel backswards and right wheel forwards, then right wheel backwards and left wheel forwards.
        """
        for _ in range(number_of_jiggles):
            self.tank_drive.on_for_degrees(left_speed=-20, right_speed=20, degrees=jiggle_degrees)
            sleep(0.1)
            self.tank_drive.on_for_degrees(left_speed=20, right_speed=-20, degrees=jiggle_degrees)
            sleep(0.1)

    def open_claw(self):
        self.claw_motor.on_to_position(speed=20, position_sp=self.CLAW_OPEN_POS)

    def close_claw(self):
        self.claw_motor.on_to_position(speed=20, position_sp=self.CLAW_CLOSED_POS)

    # 
    def deliver_ball(self, cm_amount=4):
        self.move_in_cm(cm_amount, cm_amount)
        sleep(0.5)
        self.open_claw()
        sleep(0.5)
        self.move_in_cm(-cm_amount, -cm_amount)
        sleep(0.5)
        # maybe jiggle here if the ball is stuck
        #self.perform_jiggle(2, 46)
        self.move_in_cm(cm_amount, cm_amount)
        sleep(0.5)
        self.move_in_cm(-cm_amount, -cm_amount)


'''
Write ipconfig in cmd

Wireless LAN adapter Wi-Fi:

   Connection-specific DNS Suffix  . :
   Link-local IPv6 Address . . . . . : fe80::c4d2:a884:a170:1f%15
   IPv4 Address. . . . . . . . . . . : 192.168.0.197 # replacem with this value
   Subnet Mask . . . . . . . . . . . : 255.255.255.0
   Default Gateway . . . . . . . . . : 192.168.0.1

'''
HOST = '192.168.187.245'
PORT = 12346

robot = Robot()

def execute_instruction(instr):
    cmd = instr.get("cmd")
    if cmd == "drive":
        left = instr.get("left_speed")
        right = instr.get("right_speed")
        if left is not None and right is not None:
            robot.move_forward(left, right)
        else:
            print("[CLIENT] Invalid drive command: " + str(instr))
            return False
    elif cmd == "claw":
        action = instr.get("action")
        if action == "open":
            print("[CLIENT] Opening claw")
            robot.open_claw()
        elif action == "close":
            robot.close_claw()
        else:
            print("[CLIENT] Unknown claw action: " + str(action))
            return False
    elif cmd == "jiggle":
        number_of_jiggles = instr.get("number_of_jiggles", 2)
        jiggle_degrees = instr.get("jiggle_degrees", 46)
        robot.perform_jiggle(number_of_jiggles, jiggle_degrees)
    elif cmd == "deliver":
        cm_amount = instr.get("cm_amount", 4)
        if cm_amount is not None:
            robot.deliver_ball(cm_amount)
        else:
            print("[CLIENT] Invalid deliver command: " + str(instr))
            return False
    else:
        print("[CLIENT] Unknown command: " + cmd)
        return False
    return True

def main():
    print("[CLIENT] Connecting to camera server...")
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect((HOST, PORT))
    print("[CLIENT] Connected to server.")

    try:
        while True:
            data = client.recv(1024)
            if not data:
                print("[CLIENT] No data received. Exiting.")
                break

            try:
                instruction = json.loads(data.decode())
                print("[CLIENT] Received instruction: " + str(instruction))
            except json.JSONDecodeError:
                print("[CLIENT] Failed to decode instruction.")
                break

            if execute_instruction(instruction):
                print("[CLIENT] Instruction executed.")
                client.sendall(json.dumps({"status": "done"}).encode())
            else:
                client.sendall(json.dumps({"status": "error", "msg": "invalid command"}).encode())

    except KeyboardInterrupt:
        print("[CLIENT] Interrupted.")

    finally:
        client.close()
        print("[CLIENT] Connection closed.")

if __name__ == '__main__':
    main()