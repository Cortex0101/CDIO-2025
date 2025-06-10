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

    def open_claw(self):
        self.claw_motor.on_to_position(speed=20, position_sp=self.CLAW_OPEN_POS)

    def close_claw(self):
        self.claw_motor.on_to_position(speed=20, position_sp=self.CLAW_CLOSED_POS)

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
        if action == "open":
            robot.open_claw()
        elif action == "close":
            robot.close_claw()
        else:
            print("[CLIENT] Unknown claw action: " + str(action))
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