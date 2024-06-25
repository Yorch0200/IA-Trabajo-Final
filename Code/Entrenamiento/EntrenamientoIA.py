import tensorflow as tf
from keras import layers, models

# Define el directorio de las imágenes
train_dir = 'ImagenesFinal/Entrenamiento'  # Cambia esto por la ruta a tu directorio de entrenamiento
val_dir = 'ImagenesFinal/Validacion'  # Cambia esto por la ruta a tu directorio de validación

# Cargar datos de entrenamiento y validación
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    image_size=(178, 218),  # Cambia las dimensiones de las imágenes
    batch_size=128,  # Cambiado a 64
    label_mode='int'
)

validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    val_dir,
    image_size=(178, 218),  # Cambia las dimensiones de las imágenes
    batch_size=128,  # Cambiado a 64
    label_mode='int'
)

# Definir el modelo
model = models.Sequential([
    layers.Rescaling(1./255, input_shape=(178, 218, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(train_dataset.class_names), activation='softmax')  # Número de clases
])

# Compilar el modelo
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Entrenar el modelo
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=20  # Ajusta el número de épocas según sea necesario
)

# Guardar el modelo
model.save('EntrenamientoFinal/ModeloEntrenado.h5')
