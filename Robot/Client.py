import socket
import json
from Robot import Robot  # Make sure your Robot class is in robot.py
from ev3dev2.sound import Sound

HOST = '192.168.120.245'  # Replace with actual IP of camera server
PORT = 12346

robot = Robot()
sound = Sound()

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
                sound.beep()
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
