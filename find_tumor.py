import cv2
import numpy as np

# Load the image
image_path = 'results/lungs_image_with_tumor.jpg'
image = cv2.imread(image_path)

# Check if the image has been loaded
if image is not None:
    # Convert the image to RGB (OpenCV loads images in BGR by default)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Define the range of red color in RGB
    lower_red = np.array([100, 0, 0])
    upper_red = np.array([255, 100, 100])

    # Create a mask for red color
    mask = cv2.inRange(image_rgb, lower_red, upper_red)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Assuming we are looking for the largest contour which would be our red shape
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        # Get the bounding rectangle of the largest contour
        x, y, w, h = cv2.boundingRect(largest_contour)
        red_shape_coordinates = (x, y, w, h)
    else:
        red_shape_coordinates = None
else:
    red_shape_coordinates = None

print(red_shape_coordinates)
