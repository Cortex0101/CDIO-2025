from ultralytics import YOLO
import cv2
import numpy as np
import socket
import json
import time
import math
from sklearn.decomposition import PCA

model = YOLO("ball_detect/v7/weights/best.pt")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

def draw_box(img, box):
    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
    conf = box.conf[0].cpu().item()

    # compute center
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)

    # draw rectangle
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # show confidence above the box
    cv2.putText(
            img,
            f"{conf:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

        # draw center‐point and coords
    cv2.circle(img, (cx, cy), 3, (0, 0, 255), -1)
    cv2.putText(
            img,
            f"({cx},{cy})",
            (cx + 5, cy - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
        )

def connect_to_ev3(host, port):
    HOST = host
    PORT = port

    print("[SERVER] Starting camera server...")

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((HOST, PORT))
    server.listen(1)

    print(f"[SERVER] Listening on port {PORT}... Waiting for EV3 connection.")
    conn, addr = server.accept()
    print(f"[SERVER] EV3 robot connected from {addr}")

    return conn, addr
    
def get_objects(img):
    # returns a tuple, with x,y center coordinates of the detected objects and their labels
    results = model.predict(source=img, conf=0.3, iou=0.5)
    r = results[0]
    objects = []
    for box in r.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        label = r.names[box.cls[0].item()]
        objects.append((cx, cy, label))
    
    return objects

#conn, addr = connect_to_ev3('0.0.0.0', 123456)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    img = frame.copy()

    results = model.predict(source=frame, conf=0.3, iou=0.5)
    r = results[0]

    img = frame.copy()
    # Loop over each detected box
    for box in r.boxes:
        draw_box(img, box)

    # display
    cv2.imshow("Detections + Coords", img)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    objects = get_objects(frame)

    # print x, y coordinates of the all 'white' objects
    white_balls = [obj for obj in objects if obj[2] == 'white']

    print(f"[SERVER] Detected white balls: {white_balls}")



cap.release()
cv2.destroyAllWindows()
