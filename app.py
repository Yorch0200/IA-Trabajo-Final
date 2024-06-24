from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
from PIL import Image
import io

app = Flask(__name__)

# Carga el modelo
model = tf.keras.models.load_model('Model/ModeloEntrenado.h5')

def preprocess_image(image):
    # Ajusta el tamaño de la imagen y otros preprocesamientos necesarios
    image = image.resize((178, 218))  # Ajusta al tamaño que tu modelo espera
    image = np.array(image)
    image = np.expand_dims(image, axis=0)  # Añade una dimensión para el batch
    return image

@app.route('/')
def index():
    return render_template('Index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        try:
            img = Image.open(io.BytesIO(file.read()))
            img = preprocess_image(img)
            prediction = model.predict(img)
            return jsonify({'prediction': prediction.tolist()})
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    else:
        return jsonify({'error': 'File type not allowed'}), 400

def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

if __name__ == '__main__':
    app.run(debug=True)
