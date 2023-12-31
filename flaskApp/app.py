# app.py
from flask import Flask, request, render_template, send_from_directory, url_for
from ImageProcessor import ImageProcessor  # Importuj klasę ImageProcessor
from werkzeug.utils import secure_filename
import os
import json

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'static'
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/process', methods=['POST'])
def process_image():
    data = request.json
    filename = data['filename']
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    processor = ImageProcessor(
        image_path=image_path,
        black_bg_threshold=data['black_bg_threshold'],
        threshold_decrement=data['threshold_decrement'],
        brightness_cutoff_percent = data['brightness_cutoff_percent'],
        iterations=data['iterations'],
        min_contiguous_pixels=data['min_contiguous_pixels'],
    )

    output_image = processor.output_image
    output_filename = 'processed_' + filename
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
    processor.save_result(output_image, output_path)

    return {'image_url': url_for('static', filename=output_filename)}

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    if file:
        filename = secure_filename(file.filename)
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(image_path)

        return {'image_url': url_for('static', filename=filename)}
    return {'error': 'No file uploaded'}, 400

if __name__ == '__main__':
    app.run(debug=True)
