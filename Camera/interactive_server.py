import socket
import json

HOST = '0.0.0.0'
PORT = 12345

print(f"[SERVER] Waiting for EV3 to connect on {HOST}:{PORT}...")
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind((HOST, PORT))
server.listen(1)
conn, addr = server.accept()
print(f"[SERVER] Connected by {addr}")

def send_instruction(instruction):
    conn.sendall(json.dumps(instruction).encode())
    data = conn.recv(1024)
    try:
        response = json.loads(data.decode())
    except Exception:
        response = {"status": "error", "msg": "Invalid response"}
    return response

def main():
    print("[SERVER] Ready to send commands. Type `exit` to quit.")
    while True:
        raw = input("Command (e.g. move 100 or turn -90): ").strip()
        if raw.lower() == "exit":
            break

        parts = raw.split()
        if len(parts) != 2:
            print("Invalid input. Use format: move 100 or turn -90")
            continue

        cmd, value = parts[0], parts[1]
        try:
            value = int(value)
        except ValueError:
            print("Second part must be an integer.")
            continue

        if cmd not in ["move", "turn"]:
            print("Command must be 'move' or 'turn'.")
            continue

        instr = {"cmd": cmd, "distance" if cmd == "move" else "angle": value}
        print(f"[SERVER] Sending: {instr}")
        response = send_instruction(instr)
        print(f"[EV3] Response: {response}")

    print("[SERVER] Closing connection.")
    conn.close()

if __name__ == "__main__":
    main()
