# ImageProcessor/image_processor.py
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


class ImageProcessor:
    def __init__(self, image_path, threshold_decrement, brightness_cutoff_percent, iterations, min_contiguous_pixels,
                 blur_kernel_size=(5, 5), noise_kernel_size=(10, 10),
                 lung_contour_area_threshold=7000, black_bg_threshold=110, overlay_alpha=0.3):
        self.image_path = image_path
        self.blur_kernel_size = blur_kernel_size
        self.noise_kernel_size = noise_kernel_size
        self.lung_contour_area_threshold = lung_contour_area_threshold
        self.black_bg_threshold = black_bg_threshold
        self.overlay_alpha = overlay_alpha
        self.brightness_cutoff_percent = brightness_cutoff_percent
        self.iterations = iterations
        self.min_contiguous_pixels = min_contiguous_pixels
        self.threshold_decrement = threshold_decrement
        self.image = self.load_image()
        self.background_mask = None
        self.blurred = self.apply_gaussian_blur()
        self.binary_image = self.create_dynamic_threshold_image()
        self.cleaned = self.remove_noise()
        self.contours = self.find_contours()
        self.lung_mask = self.draw_lungs_contours()
        self.colored_images = self.color_images()
        self.prepare_final_mask()
        self.overlay = self.apply_overlay()
        self.output_image = self.prepare_output()

    def load_image(self):
        image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError("Image cannot be loaded. Please check the file path.")
        return image

    def apply_background_mask(self, image):
        if self.background_mask is None:
            self.background_mask = self.create_background_mask(image)
        return cv2.bitwise_and(image, image, mask=self.background_mask)

    def create_background_mask(self, image):
        _, binary_image = cv2.threshold(image, self.black_bg_threshold, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            contour_mask = np.zeros_like(image)
            cv2.drawContours(contour_mask, [largest_contour], -1, 255, -1)
            return contour_mask
        else:
            return np.ones_like(image) * 255  # If no contours found, return a mask that doesn't mask anything

    def apply_gaussian_blur(self):
        image_masked = self.apply_background_mask(self.image)
        return cv2.GaussianBlur(image_masked, self.blur_kernel_size, 0)

    def create_dynamic_threshold_image(self):
        image_masked = self.apply_background_mask(self.blurred).copy()
        binary_image = np.zeros_like(image_masked)  # Utworzenie początkowego obrazu binarnego
        current_brightness_cutoff = self.brightness_cutoff_percent
        nonzero_indices = image_masked > 0

        for _ in range(self.iterations):

            if not np.any(nonzero_indices):  # Przerwanie, jeśli nie ma niezerowych pikseli
                break

            mean_brightness = np.mean(image_masked[nonzero_indices])
            brightness_threshold = mean_brightness * (current_brightness_cutoff / 100.0)

            binary_image[nonzero_indices] = np.where(image_masked[nonzero_indices] <= brightness_threshold, 255,
                                                     0).astype(np.uint8)

            nonzero_indices = binary_image > 0  # Aktualizacja dla kolejnej iteracji

            # Zmniejszenie progu jasności o zadany dekrement
            current_brightness_cutoff = max(current_brightness_cutoff - self.threshold_decrement, 0)

        kernel = np.ones((self.min_contiguous_pixels, self.min_contiguous_pixels), np.uint8)
        binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)

        return binary_image

    def remove_noise(self):
        return cv2.morphologyEx(self.binary_image, cv2.MORPH_OPEN, np.ones(self.noise_kernel_size, np.uint8))

    def find_contours(self):
        contours, _ = cv2.findContours(self.cleaned.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def draw_lungs_contours(self):
        lung_mask = np.zeros_like(self.image)
        for contour in self.contours:
            area = cv2.contourArea(contour)
            if area > self.lung_contour_area_threshold:
                cv2.drawContours(lung_mask, [contour], -1, (255), thickness=-1)
        lung_mask = self.apply_background_mask(lung_mask)
        return lung_mask

    def prepare_final_mask(self):
        # Znajdź kontury na masce płuc
        contours, _ = cv2.findContours(self.lung_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Posortuj kontury według powierzchni w porządku malejącym i weź dwa największe
        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]

        # Stwórz nową pustą maskę
        final_mask = np.zeros_like(self.lung_mask, dtype=np.uint8)

        # Narysuj dwa największe kontury na masce
        cv2.drawContours(final_mask, sorted_contours, -1, (255), thickness=cv2.FILLED)

        # Stwórz finalny obraz, nakładając nową maskę na obraz kolorowy z płucami
        final_lung_area = cv2.bitwise_and(self.color_images()[1], self.color_images()[1], mask=final_mask)

        self.colored_images = self.colored_images[0], final_lung_area

    def color_images(self):
        original_colored = cv2.cvtColor(self.apply_background_mask(self.image), cv2.COLOR_GRAY2BGR)
        lung_colored = cv2.cvtColor(self.apply_background_mask(self.lung_mask), cv2.COLOR_GRAY2BGR)
        lung_colored[:, :, 1:3] = 0  # Color the mask red for visibility
        return original_colored, lung_colored

    def apply_overlay(self):
        return cv2.addWeighted(self.colored_images[0], 1, self.colored_images[1], self.overlay_alpha, 0)

    def prepare_output(self):
        return np.hstack((self.colored_images[0], self.overlay))

    def save_result(self, image, output_path):
        cv2.imwrite(output_path, image)

    def show_result(self, image):
        cv2.imshow('Result', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def get_histogram(self):
        masked_image = self.apply_background_mask(self.image)
        hist = cv2.calcHist([masked_image], [0], None, [256], [0, 256])
        hist_normalized = hist.ravel() / hist.max()
        return hist_normalized

    def save_histogram(self, output_path):
        # Mask the image to focus on the area of interest
        masked_image = self.apply_background_mask(self.image)
        # Calculate the histogram
        hist = cv2.calcHist([masked_image], [0], None, [256], [0, 256])

        # Normalize the histogram
        hist_normalized = hist.ravel() / hist.sum()
        # Calculate bins center
        bins = np.arange(256)

        # Plot the histogram
        plt.figure()
        plt.title('Grayscale Histogram')
        plt.xlabel('Bin')
        plt.ylabel('Frequency')
        plt.plot(bins, hist_normalized)
        plt.xlim([0, 256])

        # Save the histogram plot
        histogram_plot_filename = 'histogram_plot.png'
        plt.savefig(os.path.join(output_path, histogram_plot_filename))
        plt.close()

        # Save the histogram data
        histogram_data_filename = 'histogram_data.txt'
        np.savetxt(os.path.join(output_path, histogram_data_filename), hist_normalized)
