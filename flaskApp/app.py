from flask import Flask, request, render_template, send_from_directory, url_for
from ImageProcessor import ImageProcessor
from werkzeug.utils import secure_filename
import os
import json

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'  # Upewnij się, że ten folder istnieje i ma odpowiednie uprawnienia
app.config['OUTPUT_FOLDER'] = 'static'
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # Opcjonalnie, aby zapobiec cachowaniu

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Strona główna z formularzem do przesyłania
@app.route('/')
def index():
    return render_template('upload.html')

# Endpoint do przetwarzania obrazu
@app.route('/process', methods=['POST'])
def process_image():
    data = request.json
    filename = data['filename']
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    # Tutaj tworzymy instancję ImageProcessor z parametrami przekazanymi przez użytkownika
    processor = ImageProcessor(
        image_path=image_path,
        use_otsu=data['use_otsu'],
        use_histogram=data['use_histogram'],
        lung_contour_area_threshold=data['lung_contour_area_threshold'],
        adaptive_threshold_block_size=data['adaptive_threshold_block_size'],
        adaptive_threshold_C=data['adaptive_threshold_C']
    )

    output_image = processor.output_image
    output_filename = 'processed_' + filename
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
    processor.save_result(output_image, output_path)

    return {'image_url': url_for('static', filename=output_filename)}

# Endpoint do przesyłania plików
@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    if file:
        filename = secure_filename(file.filename)
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(image_path)

        # Tu możesz przetworzyć obrazek z domyślnymi ustawieniami i zwrócić ścieżkę do niego
        # ...

        return {'image_url': url_for('static', filename=filename)}
    return {'error': 'No file uploaded'}, 400

if __name__ == '__main__':
    app.run(debug=True)
