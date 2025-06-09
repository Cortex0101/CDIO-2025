import socket
import json
import time
import math

class Server:
    def __init__(self):
        self.host = '0.0.0.0'
        self.port = 12346
        self.SEND_CUSTOM_INSTRUCTIONS = True

        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.bind((self.host, self.port))
        self.server.listen(1)
        print(f"[SERVER] Listening on {self.host}:{self.port}... Waiting for EV3 connection.")
        self.conn, addr = self.server.accept()

        # if send_custom_instructions is true, we will be able to simply send instruction by entering
        # them in the console, otherwise we will have to use we will run the main_loop() method
        if self.SEND_CUSTOM_INSTRUCTIONS:
            print("[SERVER] Custom instruction mode enabled. Type `exit` to quit.")
            self.custom_instruction_loop()
        else:
            print("[SERVER] Custom instruction mode disabled. Running main loop.")
            self.main_loop()

    def accept_connection(self):
        self.conn, addr = self.server.accept()
        print(f"[SERVER] EV3 robot connected from {addr}")

    def main_loop():
        while True:
             # do nothing for now
            pass

    def send_instruction(self, instruction):
        try:
            self.conn.sendall(json.dumps(instruction).encode())
            data = self.conn.recv(1024)
            response = json.loads(data.decode())
            return response
        except Exception as e:
            print(f"[SERVER] Error sending instruction: {e}")
            return {"status": "error", "msg": str(e)}

    def custom_instruction_loop(self):
        print("[SERVER] Ready to send custom instructions. Type `exit` to quit.")
        while True:
            raw = input("Command (e.g. move 10 20 or claw open or claw close): with it being 10 on left motor and 20 on right motor").strip()
            if raw.lower() == "exit":
                break

            parts = raw.split()

            cmd = parts[0]

            if cmd == "move":
                if len(parts) != 3:
                    print("Invalid move command. Use format: move <left_speed> <right_speed>")
                    continue
                try:
                    left_speed = int(parts[1])
                    right_speed = int(parts[2])
                except ValueError:
                    print("Left and right speeds must be integers.")
                    continue
                instruction = {"cmd": "drive", "left_speed": left_speed, "right_speed": right_speed}
                self.send_instruction(instruction)
            elif cmd == "claw":
                if len(parts) != 2 or parts[1] not in ["open", "close"]:
                    print("Invalid claw command. Use format: claw open or claw close")
                    continue
                action = parts[1]
                instruction = {"cmd": "claw", "action": action}
                self.send_instruction(instruction)
            else:
                print("Unknown command. Use 'move' or 'claw'.")
                continue

if __name__ == "__main__":
    server = Server()