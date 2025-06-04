from ultralytics import YOLO
import cv2
import numpy as np

# Load your trained model
model = YOLO("ball_detect/v4_s_balls_final/weights/best.pt")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run inference (returns a list of Results; we only care about the first)
    results = model.predict(source=frame, conf=0.3, iou=0.5)
    r = results[0]

    # Convert to a writable image
    img = frame.copy()

    # Loop over each detected box
    for box in r.boxes:
        # xyxy = [[x1, y1, x2, y2]]
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

    # display
    cv2.imshow("Detections + Coords", img)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
