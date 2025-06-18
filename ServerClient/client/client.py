#!/usr/bin/env python3

from ev3dev2.motor import LargeMotor, MediumMotor, OUTPUT_A, OUTPUT_B, OUTPUT_C, OUTPUT_D, MoveTank
from ev3dev2.sound import Sound
from time import sleep
import math
import socket
import json
import sys
import time

class Robot:
    WHEEL_DIAMETER = 5.5 # cm
    WHEEL_CIRCUMFERENCE = math.pi * WHEEL_DIAMETER  # cm per full wheel rotation
    AXLE_TRACK = 16.5  # cm (distance between left and right wheels)

    CLAW_OPEN_POS = 80
    CLAW_CLOSED_POS = 0
    
    def __init__(self):
        self.left_motor = LargeMotor(OUTPUT_C)
        self.right_motor = LargeMotor(OUTPUT_D)
        self.claw_motor = MediumMotor(OUTPUT_B)
        self.tank_drive = MoveTank(OUTPUT_C, OUTPUT_D)

    def move_forward(self, left_speed, right_speed):
        self.tank_drive.on(left_speed, right_speed)

    def move_forward_for_seconds(self, seconds):
        self.tank_drive.on_for_seconds(left_speed=50, right_speed=50, seconds=seconds)
    def move_backwards_for_seconds(self, seconds):
        self.tank_drive.on_for_seconds(left_speed=-50, right_speed=-50, seconds=seconds)

    def perform_jiggle(self, number_of_jiggles=2, jiggle_degrees=4):
        """
        Jiggles the robot left and right by first turning left wheel backswards and right wheel forwards, then right wheel backwards and left wheel forwards.
        """
        for _ in range(number_of_jiggles):
            self.tank_drive.on_for_degrees(left_speed=-20, right_speed=20, degrees=jiggle_degrees)
            sleep(0.1)
            self.tank_drive.on_for_degrees(left_speed=20, right_speed=-20, degrees=jiggle_degrees)
            sleep(0.1)

    def open_claw(self, speed=20):
        self.claw_motor.on_to_position(speed=speed, position=self.CLAW_OPEN_POS)

    def close_claw(self, speed=20):
        self.claw_motor.on_to_position(speed=speed, position=self.CLAW_CLOSED_POS)

    def deliver_ball(self, speed=75):
        # opens the claw, then moves forward 1 rotation with speed and then back 1 rotation
        self.open_claw(5)
        sleep(0.5)
        self.tank_drive.on_for_degrees(left_speed=speed, right_speed=speed,  degrees=90)
        sleep(0.5)
        self.tank_drive.on_for_degrees(left_speed=-10, right_speed=-10,  degrees=90)

    def emergency_stop(self):
        self.tank_drive.off()
        self.claw_motor.off()
        self.left_motor.off()
        self.right_motor.off()
        


'''
Write ipconfig in cmd

Wireless LAN adapter Wi-Fi:

   Connection-specific DNS Suffix  . :
   Link-local IPv6 Address . . . . . : fe80::c4d2:a884:a170:1f%15
   IPv4 Address. . . . . . . . . . . : 192.168.0.197 # replacem with this value
   Subnet Mask . . . . . . . . . . . : 255.255.255.0
   Default Gateway . . . . . . . . . : 192.168.0.1

'''
HOST = '192.168.0.197'
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
        speed = instr.get("speed", 20)
        if action == "open":
            print("[CLIENT] Opening claw")
            robot.open_claw(speed)
        elif action == "close":
            robot.close_claw(speed)
        else:
            print("[CLIENT] Unknown claw action: " + str(action))
            return False
    elif cmd == "jiggle":
        number_of_jiggles = instr.get("number_of_jiggles", 2)
        jiggle_degrees = instr.get("jiggle_degrees", 46)
        robot.perform_jiggle(number_of_jiggles, jiggle_degrees)
    elif cmd == "deliver":
        speed = instr.get("speed", 75)
        if speed is not None:
            robot.deliver_ball(speed)
        else:
            print("[CLIENT] Invalid deliver command: " + str(instr))
            return False
    else:
        print("[CLIENT] Unknown command: " + cmd)
        return False
    return True

def main():
    while True:
        try:
            print("[CLIENT] Connecting to camera server...")
            client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client.connect((HOST, PORT))
            print("[CLIENT] Connected to server.")

            # Use a file-like object for line-based protocol
            file = client.makefile('r')
            while True:
                line = file.readline()
                if not line:
                    print("[CLIENT] No data received. Exiting connection loop.")
                    break

                try:
                    instruction = json.loads(line)
                    print("[CLIENT] Received instruction: " + str(instruction))
                except json.JSONDecodeError:
                    print("[CLIENT] Failed to decode instruction.")
                    break

                if execute_instruction(instruction):
                    print("[CLIENT] Instruction executed.")
                    client.sendall((json.dumps({"status": "done"}) + '\n').encode())
                else:
                    client.sendall((json.dumps({"status": "error", "msg": "invalid command"}) + '\n').encode())

        except (socket.error, json.JSONDecodeError) as e:
            print("[CLIENT] Error: " + str(e))
            try:
                client.sendall((json.dumps({"status": "error", "msg": str(e)}) + '\n').encode())
            except Exception:
                pass
            print("[CLIENT] Connection error. Will retry in 3 seconds.")

        finally:
            try:
                client.close()
                robot.emergency_stop()
            except Exception:
                pass
            print("[CLIENT] Connection closed. Retrying in 3 seconds...")
            time.sleep(3)
            robot.close_claw()
            

if __name__ == '__main__':
    main()