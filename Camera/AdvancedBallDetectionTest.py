from AdvancedBallDetection import detect_balls

if __name__ == "__main__":
    import cv2

    cap = cv2.VideoCapture(0)  # Change as needed

    while True:
        # get frame from jpg at D:\CDIO25\CDIO-2025\ressources\img\48912632-2c4a-48e8-93d1-3ca11843cd76.jpg
        frame = cv2.imread(r"D:\\CDIO25\\CDIO-2025\\ressources\\img\\48912632-2c4a-48e8-93d1-3ca11843cd76.jpg")

        ball_positions = detect_balls(frame, DEBUGGING=True)
        print("Detected balls:", ball_positions)

        if ball_positions is None:
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()