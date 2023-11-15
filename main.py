import cv2
import numpy as np

class ImageProcessor:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = self.load_image()
        self.blurred = self.apply_gaussian_blur()
        self.binary_image = self.apply_threshold()
        self.cleaned = self.remove_noise()
        self.contours = self.find_contours()
        self.lung_mask = self.draw_lungs_contours()
        self.colored_images = self.color_images()
        self.overlay = self.apply_overlay()
        self.output_image = self.prepare_output()

    def load_image(self):
        image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError("Image cannot be loaded. Please check the file path.")
        return image

    def apply_gaussian_blur(self):
        return cv2.GaussianBlur(self.image, (5, 5), 0)

    def apply_threshold(self):
        _, binary_image = cv2.threshold(self.blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return binary_image

    def remove_noise(self):
        kernel = np.ones((15, 15), np.uint8)
        return cv2.morphologyEx(self.binary_image, cv2.MORPH_OPEN, kernel)

    def find_contours(self):
        contours, _ = cv2.findContours(self.cleaned.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def draw_lungs_contours(self):
        lung_mask = np.zeros_like(self.image)
        for contour in self.contours:
            area = cv2.contourArea(contour)
            if area > 5000:  # This threshold for area can be adjusted
                cv2.drawContours(lung_mask, [contour], -1, (255), thickness=-1)
        return lung_mask

    def color_images(self):
        original_colored = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)
        lung_colored = cv2.cvtColor(self.lung_mask, cv2.COLOR_GRAY2BGR)
        lung_colored[:, :, 1:3] = 0  # Color the mask red for visibility
        return original_colored, lung_colored

    def apply_overlay(self):
        return cv2.addWeighted(self.colored_images[0], 1, self.colored_images[1], 0.3, 0)

    def prepare_output(self):
        return np.hstack((self.colored_images[0], self.overlay))

    def save_result(self, output_path='xray_with_lung_overlay.jpg'):
        cv2.imwrite(output_path, self.output_image)

    def show_result(self):
        cv2.imshow('Result', self.output_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def remove_black_background(self, threshold=30, make_transparent=False):
        # Convert to a color image if necessary
        colored_image = cv2.cvtColor(self.output_image, cv2.COLOR_GRAY2BGR) if len(
            self.output_image.shape) == 2 else self.output_image.copy()

        # Thresholding to create a mask for the black background
        _, mask = cv2.threshold(cv2.cvtColor(colored_image, cv2.COLOR_BGR2GRAY), threshold, 255, cv2.THRESH_BINARY)

        if make_transparent:
            # Convert image to RGBA if transparency is needed
            colored_image = cv2.cvtColor(colored_image, cv2.COLOR_BGR2BGRA)
            colored_image[:, :, 3] = mask  # Set the alpha channel according to the mask
        else:
            # Replace black background with white
            colored_image[mask == 0] = (255, 255, 255)

        return colored_image


if __name__ == '__main__':
    processor = ImageProcessor('chest/IM-0147-0001.jpeg')
    processor.save_result()
    # processor.show_result()
