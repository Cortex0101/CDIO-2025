import cv2
import numpy as np


def select_corners(img):
    clicked_points = []
    magnifier_size = 30  # region to magnify (square of this size)
    zoom_factor = 8       # how much to zoom in

    clone = img.copy()
    display = clone.copy()

    def click_event(event, x, y, flags, param):
        nonlocal display
        if event == cv2.EVENT_LBUTTONDOWN and len(clicked_points) < 4:
            clicked_points.append((x, y))
            print(f"Point {len(clicked_points)}: ({x}, {y})")
            # Draw a small circle where clicked
            cv2.circle(display, (x, y), 5, (0, 255, 0), -1)

            if len(clicked_points) == 4:
                print("\nAll 4 points clicked:")
                for i, point in enumerate(clicked_points):
                    print(f"  Corner {i+1}: {point}")
                cv2.destroyAllWindows()

        elif event == cv2.EVENT_MOUSEMOVE:
            # Redraw original display image
            display = clone.copy()

            # Draw magnifier view
            half = magnifier_size // 2
            xmin = x - half
            xmax = x + half
            ymin = y - half
            ymax = y + half

            # Create a black canvas for the magnifier
            magnifier = np.zeros((magnifier_size, magnifier_size, 3), dtype=np.uint8)

            # Calculate valid region within the image
            valid_xmin = max(xmin, 0)
            valid_xmax = min(xmax, img.shape[1])
            valid_ymin = max(ymin, 0)
            valid_ymax = min(ymax, img.shape[0])

            # Map the valid region to the magnifier
            magnifier_ymin = max(0, -ymin)
            magnifier_ymax = magnifier_ymin + (valid_ymax - valid_ymin)
            magnifier_xmin = max(0, -xmin)
            magnifier_xmax = magnifier_xmin + (valid_xmax - valid_xmin)

            # Copy the valid region from the image to the magnifier
            magnifier[magnifier_ymin:magnifier_ymax, magnifier_xmin:magnifier_xmax] = img[valid_ymin:valid_ymax, valid_xmin:valid_xmax]

            # Zoom the magnifier
            zoomed = cv2.resize(magnifier, (magnifier_size * zoom_factor, magnifier_size * zoom_factor), interpolation=cv2.INTER_NEAREST)

            # Draw a crosshair on the zoomed image
            cv2.line(zoomed, (zoomed.shape[1] // 2, 0), (zoomed.shape[1] // 2, zoomed.shape[0]), (0, 0, 255), 1)
            cv2.line(zoomed, (0, zoomed.shape[0] // 2), (zoomed.shape[1], zoomed.shape[0] // 2), (0, 0, 255), 1)

            # Place magnifier in the top-left corner of the display
            display[0:zoomed.shape[0], 0:zoomed.shape[1]] = zoomed

        cv2.imshow("Click 4 Corners (Zoom shown top-left)", display)

    if img is None:
        raise FileNotFoundError("Could not load image")

    cv2.imshow("Click 4 Corners (Zoom shown top-left)", display)
    cv2.setMouseCallback("Click 4 Corners (Zoom shown top-left)", click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return clicked_points

def warp_image(img, corners):
    # Define real-world dimensions in cm (or any consistent unit)
    real_width = 180  # cm
    real_height = 120  # cm

    # Calculate the aspect ratio of the original image
    original_aspect_ratio = img.shape[1] / img.shape[0]

    # Define pixels per cm (scale the output size as desired)
    scale = 4.44  # e.g., 10 pixels per cm → 1800x1200 final image
    output_width = int(real_width * scale)
    output_height = int(real_height * scale)

    # Adjust the output dimensions to maintain the original aspect ratio
    if original_aspect_ratio > 1:
        output_height = int(output_width / original_aspect_ratio)
    else:
        output_width = int(output_height * original_aspect_ratio)

    # Destination points as a perfect rectangle
    dst_points = np.array([
        [0, 0],
        [output_width - 1, 0],
        [output_width - 1, output_height - 1],
        [0, output_height - 1]
    ], dtype='float32')

    # Convert corners to float32
    src_points = np.array(corners, dtype='float32')

    # Compute the perspective transform matrix
    M = cv2.getPerspectiveTransform(src_points, dst_points)

    # Perform the warp
    warped = cv2.warpPerspective(img, M, (output_width, output_height))

    return warped

def denoise_image(img): # cv.fastNlMeansDenoisingColored()
    return cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

def detect_corners(img, threshold=0.01, blockSize=2, ksize=3, k=0.04):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Use Harris corner detection
    corners = cv2.cornerHarris(gray, blockSize=blockSize, ksize=ksize, k=k)

    # Dilate to mark the corners
    corners = cv2.dilate(corners, None)

    # Threshold to get the best corners
    img[corners > threshold * corners.max()] = [0, 0, 255]

    return img

def detect_circles(gray, minRadius=0, maxRadius=0, param1=200, param2=100):
    # Apply Hough Circle Transform
    rows = gray.shape[0]
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=rows / 8,
                               param1=param1, param2=param2,
                               minRadius=minRadius, maxRadius=maxRadius)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # Draw the outer circle
            cv2.circle(gray, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # Draw the center of the circle
            cv2.circle(gray, (i[0], i[1]), 2, (0, 0, 255), 3)

    return gray

class WindowWithTrackbars:
    def __init__(self, windowName, trackbarNames, defaultValues):
        self.windowName = windowName
        self.trackbarNames = trackbarNames
        self.defaultValues = defaultValues

        cv2.namedWindow(self.windowName)
        for name, value in zip(self.trackbarNames, self.defaultValues):
            cv2.createTrackbar(name, self.windowName, value, 255, self.nothing)

        cv2.setMouseCallback(self.windowName, self.onClick)

    def nothing(self, x):
        pass

    def onClick(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"Clicked at: ({x}, {y})")
            print(f"param: {self.defaultValues}")

    def processImage(self, f):
        # Here you can implement the image processing function
        # using the current values of the trackbars
        pass

    def update(self):
        ''' Update the trackbar values and rerun the processing function if values have changed '''
        new_values = [cv2.getTrackbarPos(name, self.windowName) for name in self.trackbarNames]
        if new_values != self.defaultValues:
            self.defaultValues = new_values
            print(f"Updated values: {self.defaultValues}")
            # Here you can call your processing function with the new values
            # e.g., self.processImage(new_values)


# Load the image
imgPath = 'D:\\CDIO25\\CDIO-2025\\ressources\\img\\positives\\5.jpg'

img = cv2.imread(imgPath)
if img is None:
    raise FileNotFoundError(f"Could not load image from {imgPath}") 

# select corners
#corners = select_corners(img)
corners = [(137, 92), (129, 529), (759, 544), (735, 89)]

warped = warp_image(img, corners)
warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

# cv.bilateralFilter
warped = cv2.bilateralFilter(warped, 9, 75, 75)

# create color threshold sliders
def nothing(x):
    pass

mouse_clicked = False

cv2.namedWindow('bilateral')
cv2.createTrackbar('sigmaColor', 'bilateral', 9, 255, nothing)
cv2.createTrackbar('d', 'bilateral', 9, 255, nothing)

cv2.createTrackbar('minRadius', 'bilateral', 5, 255, nothing)
cv2.createTrackbar('maxRadius', 'bilateral', 15, 255, nothing)
cv2.createTrackbar('param1', 'bilateral', 133, 255, nothing)
cv2.createTrackbar('param2', 'bilateral', 1, 255, nothing)

new_values = [0, 0, 0, 0, 1, 1]
values = [0, 0, 0, 0, 1, 1]

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        global mouse_clicked
        mouse_clicked = True
        print(f"Clicked at: ({x}, {y})")
        print(f"param: {values[0]}, {values[1]}, {values[2]}, {values[3]}, {values[4]}, {values[5]}")
        print(f"new_param: {new_values[0]}, {new_values[1]}, {new_values[2]}, {new_values[3]}, {new_values[4]}, {new_values[5]}")

cv2.setMouseCallback('bilateral', click_event)

while True:
    # get values from sliders
    new_values[0] = cv2.getTrackbarPos('sigmaColor', 'bilateral')
    new_values[1] = cv2.getTrackbarPos('d', 'bilateral')
    new_values[2] = cv2.getTrackbarPos('minRadius', 'bilateral')
    new_values[3] = cv2.getTrackbarPos('maxRadius', 'bilateral')
    new_values[4] = cv2.getTrackbarPos('param1', 'bilateral')
    new_values[5] = cv2.getTrackbarPos('param2', 'bilateral')

    # check if values have changed & update
    if new_values[0] != values[0] or new_values[1] != values[1] or new_values[2] != values[2] or new_values[3] != values[3] or new_values[4] != values[4] or new_values[5] != values[5]:
        # was A pressed last time?
        if mouse_clicked:
            # update values
            values[0] = new_values[0]
            values[1] = new_values[1]
            values[2] = new_values[2]
            values[3] = new_values[3]
            values[4] = new_values[4]
            values[5] = new_values[5]
            # apply bilateral filter with new values
            bilateral = cv2.bilateralFilter(warped, d=values[1], sigmaColor=values[0], sigmaSpace=values[0])

            # detect circles with new values
            circles = detect_circles(bilateral, minRadius=values[2], maxRadius=values[3], param1=values[4], param2=values[5])

            # show the image
            cv2.imshow('bilateral', circles)
            # reset mouse_clicked
            mouse_clicked = False

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    


cv2.destroyAllWindows()