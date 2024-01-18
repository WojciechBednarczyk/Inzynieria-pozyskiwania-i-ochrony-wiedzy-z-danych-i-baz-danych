import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage import measure, morphology
from skimage.color import label2rgb
from skimage.measure import regionprops

def get_round_objects(image_path: str):

    original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply mask to remove black background
    mask = original_image > 0
    original_image = original_image * mask

    # Perform a Gaussian blur to the grayscale image (optional step)
    blurred = cv2.GaussianBlur(original_image, (11, 11), 0)

    # Convert the grayscale image to binary image
    _, binary = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)

    # Perform a series of erosions and dilations to remove
    # any small blobs of noise from the image.
    # binary = cv2.erode(binary, None, iterations=2)
    # binary = cv2.dilate(binary, None, iterations=4)

    # Perform closing, which is dilation followed by erosion. This is used to close small holes in the objects
    closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((15,15),np.uint8))

    # Find contours.
    contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_radius_thresh = 5
    max_radius_thresh = 500

    circular_contours = []
    for contour in contours:
        # Get the perimeter of the contour
        perimeter = cv2.arcLength(contour, True)

        # Approximate the contour with accuracy proportional to the contour perimeter
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

        # Calculating image moments to get centroid of mass
        m = cv2.moments(contour)

        # Calculating the radius of a circle with the same area as the contour
        try:
            radius = ((m['m00']/np.pi)**0.5)
        except:
            continue

        # Add to list if approximated contour has more than 8 vertices (highly likely to be circular/elliptical)
        # and radius is within the thresholds
        if len(approx) > 8 and min_radius_thresh < radius < max_radius_thresh:
            circular_contours.append(contour)

    # draw contours over original image
    cv2.drawContours(original_image, circular_contours, -1, (0, 255, 0), 3)

    # save image
    cv2.imwrite("final_image.png", original_image)

    return circular_contours

get_round_objects("results/lungs_image_with_tumor.png")