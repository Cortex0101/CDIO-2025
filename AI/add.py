# opencv open camera, when space is pressed, take a picture and save it to folder AI/images with name image_<n>.jpg where n is the number of images taken
import cv2
import os
import sys

def capture_image():
    # Create the directory if it doesn't exist
    os.makedirs("AI/images", exist_ok=True)

    # Initialize the camera
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use 0 for the default camera
    if not cap.isOpened():
        print("Error: Could not open camera.")
        sys.exit()

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

    image_count = 585


    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        cv2.imshow("Camera", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):  # Space key to capture image
            image_path = f"AI/images/image_{image_count}.jpg"
            cv2.imwrite(image_path, frame)
            print(f"Image saved: {image_path}")
            image_count += 1
        elif key == ord('q'):  # 'q' key to quit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_image()
    print("Image capture completed.")