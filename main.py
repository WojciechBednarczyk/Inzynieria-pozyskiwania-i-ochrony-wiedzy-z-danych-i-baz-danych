import cv2
import numpy as np
import os
import itertools

class ImageProcessor:
    def __init__(self, image_path, blur_kernel_size=(5, 5), noise_kernel_size=(10, 10),
                 lung_contour_area_threshold=7000, black_bg_threshold=110, use_otsu=True,
                 overlay_alpha=0.3, adaptive_threshold_block_size=301, adaptive_threshold_C=3):
        self.image_path = image_path
        self.blur_kernel_size = blur_kernel_size
        self.noise_kernel_size = noise_kernel_size
        self.lung_contour_area_threshold = lung_contour_area_threshold
        self.black_bg_threshold = black_bg_threshold
        self.use_otsu = use_otsu
        self.overlay_alpha = overlay_alpha
        self.adaptive_threshold_block_size = adaptive_threshold_block_size
        self.adaptive_threshold_C = adaptive_threshold_C

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
        return cv2.GaussianBlur(self.image, self.blur_kernel_size, 0)

    def apply_threshold(self):
        if self.use_otsu:
            _, binary_image = cv2.threshold(self.blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        else:
            binary_image = cv2.adaptiveThreshold(self.blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,
                                                 self.adaptive_threshold_block_size, self.adaptive_threshold_C)
        return binary_image

    def remove_noise(self):
        kernel = np.ones(self.noise_kernel_size, np.uint8)
        return cv2.morphologyEx(self.binary_image, cv2.MORPH_OPEN, kernel)

    def find_contours(self):
        contours, _ = cv2.findContours(self.cleaned.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def draw_lungs_contours(self):
        lung_mask = np.zeros_like(self.image)
        for contour in self.contours:
            area = cv2.contourArea(contour)
            if area > self.lung_contour_area_threshold:
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

    def remove_black_background(self, threshold=110, make_transparent=False):
        # Convert to a color image if necessary
        colored_image = cv2.cvtColor(self.output_image, cv2.COLOR_GRAY2BGR) if len(
            self.output_image.shape) == 2 else self.output_image.copy()

        # Thresholding to create a mask for the black background
        gray_image = cv2.cvtColor(colored_image, cv2.COLOR_BGR2GRAY)
        _, binary_image = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Assuming the largest contour is the main subject
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            contour_mask = np.zeros_like(gray_image)
            cv2.drawContours(contour_mask, [largest_contour], -1, 255, -1)

            if make_transparent:
                # Convert image to RGBA if transparency is needed
                colored_image = cv2.cvtColor(colored_image, cv2.COLOR_BGR2BGRA)
                # Set the alpha channel: transparent where the mask is black (background)
                colored_image[:, :, 3] = np.where(contour_mask == 0, 0, 255)
            else:
                # Replace black background with white, preserving the main subject
                for i in range(3):  # For each color channel
                    colored_image[:, :, i] = np.where(contour_mask == 0, 255, colored_image[:, :, i])
        else:
            # If no contours are found, return the original image
            return self.output_image

        return colored_image

    def save_result(self, image, output_path='xray_with_lung_overlay6.png'):
        cv2.imwrite(output_path, image)

    def show_result(self, image):
        cv2.imshow('Result', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()




def evaluate_processor(image_paths, output_dir, param_sets):
    for params in param_sets:
        for image_path in image_paths:
            processor = ImageProcessor(
                image_path=image_path,
                blur_kernel_size=params['blur_kernel_size'],
                noise_kernel_size=params['noise_kernel_size'],
                lung_contour_area_threshold=params['lung_contour_area_threshold'],
                black_bg_threshold=params['black_bg_threshold'],
                use_otsu=params['use_otsu'],
                overlay_alpha=params['overlay_alpha']
            )
            output_with_transparency = processor.remove_black_background(make_transparent=True)

            base_filename = os.path.basename(image_path)
            output_filename = f"{output_dir}/output_{base_filename}_blur{params['blur_kernel_size'][0]}_noise{params['noise_kernel_size'][0]}_threshold{params['lung_contour_area_threshold']}_bg{params['black_bg_threshold']}.png"
            processor.save_result(output_with_transparency, output_filename)


if __name__ == '__main__':
    processor = ImageProcessor(
        image_path='chest/IM-0143-0001.jpeg',
        blur_kernel_size=(5, 5),
        noise_kernel_size=(10, 10),
        lung_contour_area_threshold=7000,
        black_bg_threshold=110,
        use_otsu=False,
        overlay_alpha=0.3,
        adaptive_threshold_block_size=301,
        adaptive_threshold_C=3
    )
    output_with_transparency = processor.remove_black_background(make_transparent=True)
    processor.save_result(output_with_transparency, 'xray_with_lung_overlay.png')


    # chest_dir = 'chest_test/'
    # output_dir = 'results/'
    #
    # # Przykładowe zakresy dla każdego parametru
    # blur_kernel_sizes = [(3, 3), (5, 5)]
    # noise_kernel_sizes = [(8, 8), (10, 10)]
    # lung_contour_area_thresholds = [6000, 7000]
    # black_bg_thresholds = [90, 100, 110]
    # use_otsu_options = [True, False]
    # overlay_alphas = [0.3, 0.5]
    #
    # # Generowanie kombinacji
    # param_combinations = list(itertools.product(
    #     blur_kernel_sizes,
    #     noise_kernel_sizes,
    #     lung_contour_area_thresholds,
    #     black_bg_thresholds,
    #     use_otsu_options,
    #     overlay_alphas
    # ))
    #
    # # Ograniczenie do pierwszych 100 kombinacji
    # param_sets = [
    #     {'blur_kernel_size': combo[0], 'noise_kernel_size': combo[1], 'lung_contour_area_threshold': combo[2],
    #      'black_bg_threshold': combo[3], 'use_otsu': combo[4], 'overlay_alpha': combo[5]}
    #     for combo in param_combinations[:100]
    # ]
    # # Tutaj jest lista param_sets zdefiniowana jak wcześniej
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)
    #
    # image_paths = [os.path.join(chest_dir, f) for f in os.listdir(chest_dir) if os.path.isfile(os.path.join(chest_dir, f))]
    # evaluate_processor(image_paths, output_dir, param_sets)
