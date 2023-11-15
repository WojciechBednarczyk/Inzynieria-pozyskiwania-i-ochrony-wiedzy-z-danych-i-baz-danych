import cv2
import numpy as np
import os
import itertools
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from itertools import product


class ImageProcessor:
    def __init__(self, image_path, blur_kernel_size=(5, 5), noise_kernel_size=(10, 10),
                 lung_contour_area_threshold=7000, black_bg_threshold=110, use_otsu=True, use_histogram=False,
                 overlay_alpha=0.3, adaptive_threshold_block_size=301, adaptive_threshold_C=3):
        self.image_path = image_path
        self.blur_kernel_size = blur_kernel_size
        self.noise_kernel_size = noise_kernel_size
        self.lung_contour_area_threshold = lung_contour_area_threshold
        self.black_bg_threshold = black_bg_threshold
        self.use_otsu = use_otsu
        self.use_histogram = use_histogram
        self.overlay_alpha = overlay_alpha
        self.adaptive_threshold_block_size = adaptive_threshold_block_size
        self.adaptive_threshold_C = adaptive_threshold_C

        self.image = self.load_image()
        self.background_mask = None
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

    def apply_threshold(self):
        image_masked = self.apply_background_mask(self.blurred)
        if self.use_histogram:
            histogram_data = self.get_histogram()
            suggested_thresholds = self.analyze_histogram(histogram_data)
            if suggested_thresholds:
                # Możesz tu zastosować różne strategie wyboru wartości progu
                # Na przykład, możesz wybrać najmniejszą dolinę, największą dolinę, średnią dolinę, itp.
                # W tym przykładzie wybieramy najmniejszą dolinę jako wartość progu
                threshold_value = min(suggested_thresholds)
                _, binary_image = cv2.threshold(image_masked, threshold_value, 255, cv2.THRESH_BINARY)
            else:
                # Jeśli nie znaleziono żadnych progu, zastosuj wartość domyślną lub inny sposób progowania
                _, binary_image = cv2.threshold(image_masked, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif self.use_otsu:
            _, binary_image = cv2.threshold(image_masked, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            binary_image = cv2.adaptiveThreshold(image_masked, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                                 cv2.THRESH_BINARY, self.adaptive_threshold_block_size,
                                                 self.adaptive_threshold_C)
        return binary_image

    def analyze_histogram(self, histogram_data):
        # Przyjmijmy, że ta funkcja zwraca listę sugerowanych wartości progowych opartych na histogramie
        # Na przykład może zwracać doliny między szczytami jako dobre punkty progowe
        # Zaimplementujmy tę funkcję zgodnie z wcześniejszą dyskusją
        peaks, _ = find_peaks(histogram_data, height=0.001)
        thresholds = []
        for peak in peaks:
            if histogram_data[peak] > 0.005:  # considering significant peaks only
                valley = np.argmin(histogram_data[peak:peak + 50]) + peak
                thresholds.append(valley)
        return thresholds

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

    def save_result_full_name(self, image, output_path):
        # Extract the filename and extension from the image path
        filename, ext = os.path.splitext(os.path.basename(self.image_path))

        # Create a new filename with the parameters
        new_filename = f"{filename}_blur{self.blur_kernel_size}_noise{self.noise_kernel_size}_contour{self.lung_contour_area_threshold}_bg{self.black_bg_threshold}_otsu{self.use_otsu}_alpha{self.overlay_alpha}_block{self.adaptive_threshold_block_size}_C{self.adaptive_threshold_C}{ext}"

        # Save the image with the new filename
        cv2.imwrite(os.path.join(os.path.dirname(output_path), new_filename), image)

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
