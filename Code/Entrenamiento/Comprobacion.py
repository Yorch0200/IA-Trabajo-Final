import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

# Cargar el modelo completo sin recompilar
model = load_model('EntrenamientoFinal/ModeloEntrenado.h5', compile=False)

# Función para procesar la imagen
def preprocess_image(img_path, target_size):
    img = Image.open(img_path).convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalizar la imagen
    return img_array

# Características del modelo
features = [
    '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Bags_Under_Eyes', 'Bald', 'Bangs',
    'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Brown_Hair',
    'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee',
    'Gray_Hair', 'High_Cheekbones', 'Male', 'Mustache', 'Narrow_Eyes',
    'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline',
    'Rosy_Cheeks', 'Sideburns', 'Straight_Hair', 'Wavy_Hair', 'Young'
]

# Ruta de la imagen a predecir
img_path = 'Imagen_Compobacion.jpg'

# Procesar la imagen
img_array = preprocess_image(img_path, target_size=(178, 218))  # Ajusta el tamaño objetivo según tu modelo

# Hacer la predicción
predictions = model.predict(img_array)

# Imprimir las predicciones (la matriz completa)
print('Predicciones:', predictions)

# Decodificar las predicciones usando las características
decoded_predictions = {feature: round(pred, 4) for feature, pred in zip(features, predictions[0])}

# Obtener las 5 características con los mayores valores predichos
sorted_predictions = sorted(decoded_predictions.items(), key=lambda item: item[1], reverse=True)[:9]

# Imprimir las 5 características con mayor valor
print('Características con mayor valor:')
for feature, value in sorted_predictions:
    print(f'{feature}: {value}')
