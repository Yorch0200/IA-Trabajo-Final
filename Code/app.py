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
    return render_template('index.html')

def obtenervalor(prediction):
    if isinstance(prediction, np.ndarray):
        primervalor = float(prediction[0])
    elif isinstance(prediction, list):
        primervalor = float(prediction[0])
    else:
        primervalor = prediction

    return primervalor

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if file:
        img = Image.open(io.BytesIO(file.read()))
        img = preprocess_image(img)
        prediction = model.predict(img)

        predicciones = prediction.tolist()
        

        top = predicciones
        return jsonify({'prediction': obtenervalor(predicciones)})

if __name__ == '__main__':
    app.run(debug=True)
