import cv2
import numpy as np

# Load the image of the lungs
img = cv2.imread('results/detected_circular_shapes.png')

# Convert to grayscale.
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Blur using 3 * 3 kernel.
gray_blurred = cv2.blur(gray, (3, 3))

# Apply Hough transform on the blurred image.
detected_circles = cv2.HoughCircles(gray_blurred,
                                    cv2.HOUGH_GRADIENT, 1, 100, param1=30,
                                    param2=60, minRadius=30, maxRadius=60)

# Draw circles that are detected.
if detected_circles is not None:
    # Convert the circle parameters a, b and r to integers.
    detected_circles = np.uint16(np.around(detected_circles))

    for pt in detected_circles[0, :]:
        a, b, r = pt[0], pt[1], pt[2]

        # Draw the circumference of the circle.
        cv2.circle(img, (a, b), r, (0, 255, 0), 2)

        # Draw a small circle (of radius 1) to show the center.
        cv2.circle(img, (a, b), 1, (0, 0, 255), 3)

# Show the image
cv2.imshow("Detected Circle", img)
cv2.waitKey(0)  # Wait for key press to close the window

# Save the image after drawing all circles
cv2.imwrite("results/Detected_Circles.jpg", img)

# Close all windows
cv2.destroyAllWindows()
