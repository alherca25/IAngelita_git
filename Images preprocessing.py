'''
La idea de este proyecto es generqar una IA que sea capaz de transcribir un texto escrito ne ordenador a la caligrafía de la abuela
La tarea principal es redactar el código que trate als imágenes del texto de la abuela, separar los caracteres para poder emplearlos 
como elementos de entrenamiento y ser capaz de relacionar las letras del alfabeto en tipo string con las imágenes de las letras de la 
abuela.
'''
#%%
import datetime
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
from PIL import Image 
import os

try:
    import AbuFunctions as abu
except:
    import sys
    sys.path.append('./functions')
    import AbuFunctions as abu
'''
Lo conveniente sería tener un código que lea las imágenes, separe los caracteres y los deje almacenados en una carpeta a parte. De esta
manera, tendría una base de datos con las imágenes almacenadas y etiquetadas para poder emplearlas en el entrenamiento de la IA.
'''
# Obtenemos la fecha actual y generamos una lista donde almacenar las imágenes de hoy por si acaso
date_today = datetime.date.today() 
todays_images = []

# Definimos el directorio de las imágenes iniciales
initial_images_path = r'./Initial images/' + str(date_today)

# Definimos el directorio de las imágenes etiquetadas
labeled_images_path = r'./Labeled images/' + str(date_today)

# Definimos el directorio de las imágenes editadas
normalized_images_path = r'./Normalized images/' + str(date_today)

# Obtenemos los nombres de las imágenes en la carpeta seleccionada
name_files = os.listdir(initial_images_path)
for name in name_files:
    if len(name.split('.')) != 2:
        name_files.remove(name)
print(f'Hay {len(name_files)} imágenes en la carpeta seleccionada')

# Generamos una lista donde almacenaremos las imágenes de la carpeta y, otra lista, donde almacenaremos los nombres de cada imagen
initial_images_list = []
word_list = []

# Recorremos las imágenes de la carpeta y las almacenamos en la lista junto con sus nombres
for file in name_files:
    image_path = str(initial_images_path + '\\' + file)
    image = Image.open(image_path).convert('L')
    initial_images_list.append(np.array(image))
    word_list.append(file.split('.')[0])
print(f'Los nombres de las imágenes cargadas son: {word_list}')
print('Imágenes y nombres cargados con éxito')

#%%
# Visualizamos la primera imagen de la lista
plt.figure(dpi = 130)
plt.imshow(initial_images_list[0], cmap = 'gray')
plt.grid('on', color = 'r')
plt.show()

.#%%
# Actualmente se debe elegir la imagen a recortar, pero se puede automatizar para que se recorten todas las imágenes
# Esto puede hacerse con un bucle for recorriendo todas las imágenes
''' En caso de quere modificar una sola imagen se tiene que descomentar estas líneas y seleccionar la imagen concreta
val = 0
chomped_images = abu.chompIms(initial_images_list[i], len(word_list[i]) - 2)
for cimage in chomped_images:
    plt.imshow(cimage, cmap = 'gray')
    plt.show()
    todays_images.append(cimage) 

abu.saveIms(chomped_images, word_list[val], labeled_images_path, normalized = False)
abu.saveIms(chomped_images, word_list[val], normalized_images_path, normalized = True)    
'''
for i in range(len(word_list)):
    chomped_images = abu.chompIms(initial_images_list[i], len(word_list[i]) - 2)
    for cimage in chomped_images:
        plt.imshow(cimage, cmap = 'gray')
        plt.show()
        todays_images.append(cimage)
    abu.saveIms(chomped_images, word_list[i], labeled_images_path, normalized = False)
    abu.saveIms(chomped_images, word_list[i], normalized_images_path, normalized = True)

#%%
val = 0
folder_name = word_list[val]
labeled_images_list = []
labeled_files = os.listdir(labeled_images_path + f'\\{folder_name}')
print(labeled_files)
for file in labeled_files:
    image_path = str(labeled_images_path + f'\\{folder_name}\\' + file)
    image = Image.open(image_path).convert('L')
    labeled_images_list.append(np.array(image))
print('Imágenes y nombres cargados con éxito')

abu.saveIms(labeled_images_list, word_list[val], normalized_images_path, normalized = True)
#%%
# Entramos en la carpeta de las imágenes normalizadas y las reorganizamos en la carpeta 'Final images'
final_images_path = r'./Final images'
final_images_list = []
final_images_names = []

normalized_images_path = r'./Normalized images'
normalized_files = os.listdir(normalized_images_path)

for norm_file in normalized_files:
    intern_path = normalized_images_path + rf'/{norm_file}'
    intern_files = os.listdir(intern_path)
    print(f'Los archivos dentro de la carpeta {norm_file} son: {intern_files}')

    for file in intern_files:
        intern_files_path = intern_path + rf'/{file}'
        for image_name in os.listdir(intern_files_path):
            image_path = intern_files_path + rf'/{image_name}'
            image = Image.open(image_path).convert('L')
            final_images_list.append(np.array(image))
            final_images_names.append(image_name.split('-')[0])
        print(f'Las imágenes de la carpeta {file} han sido cargadas con éxito')
    print(f'Las imágenes dentro de la carpeta {norm_file} han sido cargadas con éxito')

# Creamos un DataFrame con las imágenes finales, sus nombres y el conteo
final_images = pd.DataFrame({'Images': final_images_list, 'Name': final_images_names}).sort_values(by = 'Name')
#%% Almacenamos las imágenes en una carpeta con el nombre de la letra y la numeración de cada imagen
# Obtenemos los nombres únicos de las imágenes
unique_names = final_images['Name'].unique()

# Recorremos todas las imágenes almacenadas en la lista final_images_list y las almacenamos reetiquetadas numéricamente
for name in unique_names:
    # Extraemos las imágenes que concuerden con la variable 'name'
    images_list = final_images[final_images['Name'] == name]['Images'].tolist()

    # Generamos el direcctorio donde almacenar las imágenes
    general_path = final_images_path + f'/{name}'

    # Comprobamos si existe tal directorio y, en caso de no hacerlo, generamos la carpeta
    if not os.path.exists(general_path):
        os.makedirs(general_path)

    # Recorremos las imágenes de cada letra y almacenamos todas aquellas coincidentes con la numeración pertinente como nombre    
    for n, image in enumerate(images_list):
        # Definimos el path de la imagen
        image_path = general_path + f'/{n}.jpeg'
        
        # Almacenamos la imagen
        plt.imsave(image_path, image, cmap = 'gray')