'''
En este coódigo pretendo desarrollar la arquitectura de la red neuronal capaz de leer texto escrito a ordenador en formato 
string para convertirlo en imágenes del texto redactado con la ortografía de la abuela
'''
#%%
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Conv2D,  Dense,  Dropout, Flatten, LeakyReLU, MaxPooling2D, BatchNormalization, Conv2DTranspose, Reshape
import os
import pandas as pd
from PIL import Image

# Definimos los directorios de los cuales extraer las imágenes preprocesadas
images_path = r'./Final images'
letters_list = os.listdir(images_path)

# Cargamos las imágenes junto con sus etiquetas
images_list = []
names_list = []
shapes_list = []
for letter in letters_list:
    # Generamos el path de la carpeta de la letra
    letter_path = images_path + f'/{letter}'
    for num in os.listdir(letter_path):
        # Generamos el path de la imagen
        image_path = letter_path + f'/{num}'

        # Cargamos la imagen
        image = plt.imread(image_path)

        # Almacenamos la imagen
        images_list.append(image)

        # Alamcenamos el nombre de la imagen
        names_list.append(letter.split('_')[1])

        # Almacenamos las dimensiones de la imagen
        shapes_list.append(image.shape)
shapes_arr = np.array(shapes_list)

# Creamos un mapeo de caracteres a índices
full_chars = "abcdefghijklmnñopqrstuvwxyzABCDEFGHIJKLMNÑOPQRSTUVWXYZ0123456789 "
chars = ''.join(np.unique(names_list))
char_to_index = {char: index for index, char in enumerate(chars)}
index_to_char = {index: char for char, index in char_to_index.items()}

# Convertir etiquetas a one-hot encoding
encoded_names = tf.keras.utils.to_categorical([char_to_index[char] for char in names_list])

#%%
# Adecuamos las imágenes para que todas tengan el mismo tamaño
def resize_images_from_list(image_list, target_size):
    resized_images = []  # Lista para almacenar las imágenes redimensionadas
    
    for img in image_list:
        # Asegúrate de que la imagen esté en el formato correcto
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)  # Convierte la imagen de array a objeto PIL
        
        # Convertir imagen a escala de grises
        img = img.convert("L")

        # Redimensionar manteniendo la relación de aspecto
        img.thumbnail((target_size, target_size))
        
        # Crear una nueva imagen con fondo blanco
        new_img = Image.new("L", (target_size, target_size), 255)
        
        # Pegar la imagen redimensionada en el centro
        new_img.paste(img, ((target_size - img.size[0]) // 2,
                            (target_size - img.size[1]) // 2))
        
        # Convertir de nuevo a array si es necesario
        resized_image = np.array(new_img)  # Convertir la imagen redimensionada a array
        #resized_image = np.expand_dims(resized_image, axis=-1)  # Agregar una dimensión para el canal
        resized_images.append(resized_image / 255)  # Añadir la imagen redimensionada a la lista
    
    return resized_images

# Definimos el tamaño al que redimensionar las imágenes
target_size = 128  
resized_images = np.array(resize_images_from_list(images_list, target_size))
resized_images = resized_images.reshape(-1, target_size, target_size, 1)
#%%

def generate_synthetic_data(images, names, num_images):
    # Lista para almacenar las imágenes generadas
    generated_images = []
    generated_names = []
    
    # Generar imágenes sintéticas
    for _ in range(num_images):
        # Seleccionar una imagen al azar
        idx = np.random.choice(len(images))
        img = images[idx]
        name = names[idx]
        
        # Aplicar una transformación a la imagen
        transformed_image = apply_transformation(img)
        
        # Añadir la imagen transformada a la lista
        generated_images.append(transformed_image)
        generated_names.append(name)
    
    return np.array(generated_images), np.array(generated_names)

def apply_transformation(image):
    # Rotar la imagen aleatoriamente entre -5º y 5º
    angle = np.random.uniform(-5, 5)
    rotated_image = tf.keras.preprocessing.image.random_rotation(image, angle, row_axis=0, col_axis=1, channel_axis=2)
    
    # Cambiar la nitidez de la imagen
    sharpened_image = tf.keras.preprocessing.image.random_shear(rotated_image, intensity=0.2, row_axis=0, col_axis=1, channel_axis=2)
    
    return sharpened_image

# Obtenemos imágenes sintéticas
new_images, new_names = generate_synthetic_data(resized_images, encoded_names, 1000)

# Concatenamos las imágenes sintéticas con las originales
fullstack_images = np.concatenate([resized_images, new_images])
fullstack_names = np.concatenate([encoded_names, new_names])
#%%
# Separamos las imágenes en entrenamiento y test
train_names, test_names, train_images, test_images = train_test_split(fullstack_names, fullstack_images, test_size = 0.2, random_state = 42, stratify = fullstack_names)
train_names, val_names, train_images, val_images = train_test_split(train_names, train_images, test_size = 0.2, random_state = 42, stratify = train_names)

# Dimensión del vector de entrada 
input_dim = train_names[0].shape[0]  

# Obtenemos las dimensiones de las imágenes de salida
output_shape = train_images[0].shape

#%%
# Paso 2: Creación del modelo
'''
El one hot encoding es muy buena idea, de esa manera se puede hacer una capa input de num_clases neuronas, 
pero tengo que cambiar el modelo para que partiendo de este one-hot llegue a las imágenes que yo quiero
En caso de que introducir una capa que diferencie el one-hot puedo meter el string directamente como capa 
densa de 1 neurona y que empiece el proceso hacia las imágenes
'''

# Creamos el modelo
input_layer = Input(shape=(input_dim,))
x = Dense(128, activation = 'relu')(input_layer)
x = Dropout(0.25)(x)
x = Dense(256, activation = 'relu')(x)
x = Dropout(0.25)(x)
x = Dense(512, activation = 'relu')(x)
x = Dropout(0.25)(x)
x = Dense(1024, activation = 'relu')(x)
x = Reshape((8, 8, 16))(x)
x = Conv2DTranspose(64, (3, 3), strides = (2, 2), padding = 'same', activation = 'relu')(x)
x = Dropout(0.25)(x)
x = Conv2DTranspose(32, (3, 3), strides = (2, 2), padding = 'same', activation = 'relu')(x)
x = Dropout(0.25)(x)
x = Conv2DTranspose(16, (3, 3), strides = (2, 2), padding = 'same', activation = 'relu')(x)
x = Dropout(0.25)(x)
x = Conv2DTranspose(8, (3, 3), strides = (2, 2), padding = 'same', activation = 'relu')(x)
x = Dropout(0.25)(x)
output_layer = Conv2D(1, (3, 3), activation = 'sigmoid', padding = 'same')(x)

model = Model(inputs = input_layer, outputs = output_layer)

# Compilamos el modelo
model.compile(optimizer = 'adam', 
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])

# Visualizamos la arquitectura del modelo
model.summary()

# Entrenamos el modelo
epochs = 25
batch = 32
history = model.fit(train_names, train_images, 
                    epochs = epochs,
                    batch_size = batch, 
                    verbose = 1,
                    shuffle = True,
                    validation_data = [val_names, val_images])

# Obtenemos los datos del entrenamiento 
accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

# Graficamos la evolución de la precisión y la pérdida
fig, (ax1, ax2) = plt.subplots(2, 1, sharex = True, figsize = (12, 8), dpi = 130)
fig.tight_layout()

# Precisión
ax1.plot(accuracy)
ax1.plot(val_accuracy)
ax1.set_title("Evolution of the model's accuracy")
ax1.set_ylabel('Accuracy')
ax1.legend(['Training', 'Validation'], loc = 'lower right')
ax1.grid()
        
# Pérdida
ax2.plot(loss)
ax2.plot(val_loss)
ax2.set_title("Evolution of the model's loss")
ax2.set_ylabel('Loss')
ax2.set_xlabel('Epoch')
ax2.legend(['Training', 'Validation'], loc = 'upper right')
ax2.grid()
ax2.set_xticks(np.arange(0, int(epochs + 1), 20))
plt.show()
#%%
# Ejemplo de uso
input_text = 'folio'

# Generamos el bucle para predecir las imágenes de las letras
handwritten_images = []
for letter in input_text:
    # Codificamos la letra introducida
    coded_array = np.zeros(len(chars))
    coded_array[char_to_index[letter]] = 1
    
    # Predecimos la imagen perteneciente a la letra introducida
    handwritten_image = model.predict(np.expand_dims(coded_array, axis=0))
    handwritten_image = np.round(handwritten_image * 255)
    
    # Mostramos la imagen obtenida
    plt.imshow(handwritten_image[0, :, :, 0])
    plt.show()

    # Almacenamos la imagen en una lista
    handwritten_images.append(handwritten_image)
