import cv2
import numpy as np

# Load the image of the lungs
# img = cv2.imread('results/final_image_tumor.jpg')
# img = cv2.imread('results/lungs_image_with_tumor.png')

# # Convert to grayscale.
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# # Blur using 3 * 3 kernel.
# gray_blurred = cv2.blur(gray, (3, 3))

# # Apply Hough transform on the blurred image.
# detected_circles = cv2.HoughCircles(gray_blurred,
#                                     cv2.HOUGH_GRADIENT_ALT, 2, 35, param1=400,
#                                     param2=0.8, minRadius=45, maxRadius=55)
# Load the image of the lungs
# img = cv2.imread('results/final_image_tumor.jpg')
img = cv2.imread('results/lungs_image_with_tumor.png')

# Convert to grayscale.
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Blur using 3 * 3 kernel.
gray_blurred = cv2.medianBlur(gray, 5)

# Apply Hough transform on the blurred image.
detected_circles = cv2.HoughCircles(gray_blurred,
                                    cv2.HOUGH_GRADIENT, 1, 10000, param1=200,
                                    param2=0.6, minRadius=45, maxRadius=55)

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
