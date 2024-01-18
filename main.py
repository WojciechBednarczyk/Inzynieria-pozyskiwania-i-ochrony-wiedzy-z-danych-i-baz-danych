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
        self.final_lung_area = self.prepare_final_mask()

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
        return final_lung_area

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

    def add_tumor_shape_to_final_mask(self, tumor_size):
        # Utwórz nową maskę na podstawie obecnej finalnej maski
        mask_with_tumor = np.copy(self.final_lung_area)

        # Znajdź kontury na masce płuc
        contours, _ = cv2.findContours(self.lung_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Posortuj kontury według powierzchni w porządku malejącym i weź dwa największe
        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]

        # Wybierz jeden z dwóch największych konturów
        selected_contour = sorted_contours[0] if len(sorted_contours) > 0 else None

        if selected_contour is not None:
            while True:
                # Losuj punkt wewnątrz konturu
                random_point_x = np.random.randint(min(selected_contour[:, :, 0])[0], max(selected_contour[:, :, 0])[0])
                random_point_y = np.random.randint(min(selected_contour[:, :, 1])[0], max(selected_contour[:, :, 1])[0])
                if cv2.pointPolygonTest(selected_contour, (random_point_x, random_point_y), False) == 1:
                    break

        # Oblicz współrzędne punktów dla kształtu guza koło-podobnego
        tumor_points = []
        deformation_factor = 0.1
        number_of_points = 20
        for i in range(number_of_points):  # Ilość punktów - dostosuj według potrzeb
            angle = 2 * np.pi * i / number_of_points
            radius = tumor_size + np.random.uniform(-deformation_factor * tumor_size, deformation_factor * tumor_size)
            x = int(random_point_x + radius * np.cos(angle))
            y = int(random_point_y + radius * np.sin(angle))
            tumor_points.append((x, y))

            # Narysuj nieregularny kształt guza na masce
            cv2.fillPoly(mask_with_tumor, [np.array(tumor_points)], color=(140, 140, 140))

        return mask_with_tumor

    def remove_blue(self, image):
        # Zero out the blue channel
        image[:, :, 0] = 0
        return image

    def remove_red(self, image):
        # Zero out the blue channel
        image[:, :, 2] = 0
        return image

    def remove_specific_color(self, image, target_color, replacement_color=(0, 0, 0)):
        # Znajdź piksele, które pasują do określonego koloru
        mask = cv2.inRange(image, target_color, target_color)

        # Zamień te piksele na wybrany kolor
        image[mask > 0] = replacement_color

        return image

    def retain_specific_color(self, image, target_color, replacement_color=(0, 0, 0)):
        # Znajdź piksele, które pasują do określonego koloru
        mask = cv2.inRange(image, target_color, target_color)

        # Utwórz obraz wynikowy, który początkowo jest cały zastąpiony kolorem zastępczym
        result = np.full(image.shape, replacement_color, dtype=np.uint8)

        # Kopiuj piksele pasujące do maski na obraz wynikowy
        result[mask > 0] = image[mask > 0]

        return result


if __name__ == '__main__':
    # Zakładamy, że ścieżka do obrazu rentgenowskiego to 'lung_xray.jpg'
    image_path = 'chest/IM-0140-0001.jpeg'

    image = cv2.imread(image_path)

    # Tworzenie instancji klasy ImageProcessor z określonymi parametrami
    processor = ImageProcessor(image_path,
                               black_bg_threshold=60,
                               brightness_cutoff_percent=100,
                               threshold_decrement=5,
                               iterations=1,
                               min_contiguous_pixels=60)

    # Przetwarzanie obrazu
    processed_image = processor.output_image

    # Dodanie kształtu guza do finalnej maski
    tumor_size = 30  # Rozmiar guza

    # generowanie guza
    final_lung_area_with_tumor = processor.add_tumor_shape_to_final_mask(tumor_size)

    # remove blue from final_lung_area_with_tumor // przygotowanie maski guza
    # tumor_area = processor.remove_blue(final_lung_area_with_tumor)
    # tumor_area = processor.remove_specific_color(final_lung_area_with_tumor, (140, 140, 140), (0, 0, 0))
    tumor_area = processor.retain_specific_color(final_lung_area_with_tumor, (140, 140, 140), (0, 0, 0))


    # prepare lung are
    lung_area = processor.remove_red(processor.final_lung_area)

    # lung_area = processor.remove_specific_color(final_lung_area_with_tumor, (140, 140, 140), (0, 0, 0))


    def combine_two_images(image1, image2):
        return cv2.addWeighted(image1, 1, image2, 0.3, 0)


    final_image_tumor = combine_two_images(image, tumor_area)
    final_image_lung = combine_two_images(image, lung_area)


    def cut_out_lung_from_image(image, mask):
        # Ensure the mask is a single-channel binary image of type uint8
        if len(mask.shape) > 2:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)[1].astype(np.uint8)

        # Resize mask to match the image size if they are different
        if mask.shape[:2] != image.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

        return cv2.bitwise_and(image, image, mask=mask)


    processor.save_result(final_image_tumor, 'results/final_image_tumor.jpg')
    processor.save_result(final_image_lung, 'results/final_image_lung.jpg')

    lungs_image = cut_out_lung_from_image(image, lung_area)

    processor.save_result(lungs_image, 'results/lungs_image.jpg')

    lungs_image_with_tumor = combine_two_images(lungs_image, tumor_area)

    processor.save_result(lungs_image_with_tumor, 'results/lungs_image_with_tumor.png')

    # Zapisanie finalnej maski z dodanym guzem
    processor.save_result(tumor_area, 'results/aaaprocessed_lung_xray_with_tumor.jpg')

    processor.save_result(processed_image, 'results/aaabbbprocessed_lung_xray.jpg')
