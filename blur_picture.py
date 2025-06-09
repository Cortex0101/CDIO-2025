
import cv2
import numpy as np
import os
import sys

def blur_image(image_path, output_path, blur_strength=5):
    """
    Apply Gaussian blur to an image and save the result.

    :param image_path: Path to the input image.
    :param output_path: Path to save the blurred image.
    :param blur_strength: Strength of the Gaussian blur (default is 5).
    """
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        print("Could not read the image.")
        sys.exit()

    # Apply Gaussian blur
    blurred_img = cv2.GaussianBlur(img, (blur_strength, blur_strength), 0)

    # Save the blurred image
    cv2.imwrite(output_path, blurred_img)
    print(f"Blurred image saved to {output_path}")

# for all images in the AI/images directory
if not os.path.exists("AI/blurred_images"):
    os.makedirs("AI/blurred_images")
with os.scandir("AI/images") as entries:
   file_count = sum(1 for entry in entries if entry.is_file())
for i in range(file_count):
    image_path = f"AI/images/image_{i}.jpg"
    output_path = f"AI/blurred_images/blurred_image_{i}.jpg"
    if os.path.exists(image_path):
        blur_image(image_path, output_path, blur_strength=15)
    else:
        print(f"Image {image_path} does not exist.")    
#blur_image("AI/images/image_0.jpg", "AI/blurred_images/blurred_image_0.jpg", blur_strength=15)