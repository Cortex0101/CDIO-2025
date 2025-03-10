#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

// Initial HSV range values
int h_min = 0, s_min = 0, v_min = 240;
int h_max = 90, s_max = 25, v_max = 255;

// Callback function for trackbars
void on_trackbar(int, void*) {}

int main()
{
    // Open the default camera
    VideoCapture cap(0, cv::CAP_DSHOW);
    if (!cap.isOpened()) {
        cout << "Error: Camera not accessible!" << endl;
        return -1;
    }

    // Create a window for trackbars
    namedWindow("HSV Trackbars", WINDOW_AUTOSIZE);

    // Create trackbars to adjust HSV values dynamically
    createTrackbar("H Min", "HSV Trackbars", &h_min, 179, on_trackbar);
    createTrackbar("H Max", "HSV Trackbars", &h_max, 179, on_trackbar);
    createTrackbar("S Min", "HSV Trackbars", &s_min, 255, on_trackbar);
    createTrackbar("S Max", "HSV Trackbars", &s_max, 255, on_trackbar);
    createTrackbar("V Min", "HSV Trackbars", &v_min, 255, on_trackbar);
    createTrackbar("V Max", "HSV Trackbars", &v_max, 255, on_trackbar);

    Mat frame, hsv, mask;

    while (true) {
        cap >> frame; // Capture frame
        if (frame.empty()) {
            cout << "Error: Empty frame!" << endl;
            break;
        }

        // Convert frame to HSV color space
        cvtColor(frame, hsv, COLOR_BGR2HSV);

        // Get the dynamically adjusted HSV values from trackbars
        Scalar lowerColor(h_min, s_min, v_min);
        Scalar upperColor(h_max, s_max, v_max);

        // Apply threshold to detect the selected color
        inRange(hsv, lowerColor, upperColor, mask);

        // Find contours of detected objects
        vector<vector<Point>> contours;
        vector<Vec4i> hierarchy;
        findContours(mask, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        // Process contours
        for (size_t i = 0; i < contours.size(); i++) {
            double area = contourArea(contours[i]);
            if (area > 500) {  // Ignore small objects
                Moments M = moments(contours[i]);

                if (M.m00 != 0) {  // Prevent division by zero
                    int cx = int(M.m10 / M.m00); // X coordinate of center
                    int cy = int(M.m01 / M.m00); // Y coordinate of center

                    cout << "Object Center: (" << cx << ", " << cy << ")" << endl;

                    // Draw bounding box
                    Rect boundingBox = boundingRect(contours[i]);
                    rectangle(frame, boundingBox, Scalar(0, 255, 0), 2);

                    // Draw center point
                    circle(frame, Point(cx, cy), 5, Scalar(0, 0, 255), -1);

                    // Display coordinates on the frame
                    putText(frame, "X: " + to_string(cx) + " Y: " + to_string(cy),
                        Point(cx + 10, cy - 10), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 2);
                }
            }
        }

        // Show the output frames
        imshow("Original Frame", frame);
        imshow("Thresholded Mask", mask);

        // Press 'Esc' to exit
        if (waitKey(30) == 27) {
            break;
        }
    }

    cap.release();
    destroyAllWindows();
    return 0;
}
