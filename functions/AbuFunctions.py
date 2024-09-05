'''
En este código se almacenan las funciones empleadas en el archivo 'Main code'
'''

import matplotlib.pyplot as plt
import numpy as np
import os

def chompIms(image, num_letters):
    '''
    Esta función se encarga de recortar las imágenes para devolver únicamente los caracteres.
    '''
    # Mostramos la imagen por si no se conocen las coordenadas de antemano
    plt.figure(dpi = 130)
    plt.imshow(image, cmap = 'gray')
    plt.grid('on', color = 'r')
    plt.show()

    # Creamos una lista para almacenar las imágenes recortadas
    chomped_images_list = []

    # Realizamos el recorte de las imágenes num veces por cada caracter que aparece en la imagen
    for n in range(num_letters):
        print(f'\nLetra número {n}')

        # Solicitamos las coordenadas de la esquina inferior izquierda de la letra en cuestión
        print('Esquina inferior izquierda: x, y')
        inf_izq = input().split(', ')
        print(inf_izq)
        izq_coor = int(inf_izq[0])
        inf_coor = int(inf_izq[1])

        # Repetimos el proceso con la esquina superior derecha para poder determinar el rectángulo ocupado por la letra
        print('Esquina superior derecha: x, y')
        sup_der = input().split(', ')
        print(sup_der)
        der_coor = int(sup_der[0])
        sup_coor = int(sup_der[1])  

        # Recortamos la imagen para obtener únicamente el caracter
        chomped_image = image[sup_coor:inf_coor, izq_coor:der_coor]

        # Almacenamos la imagen en una lista para el post-procesado
        chomped_images_list.append(chomped_image)
    return chomped_images_list


def saveIms(images_list, full_word_string, path, normalized = False):
    '''
    Esta función se encarga de almacenar las imágenes recortadas en una carpeta a parte.
    '''
    # Separamos el nombre de la palabra y el número de la imagen
    word_string = full_word_string.split('-')[0]
    num_string = full_word_string.split('-')[1]

    # Creamos la carpeta donde almacenar las imágenes
    general_path = path + f'\\{full_word_string}'
    if not os.path.exists(general_path):
        os.makedirs(general_path)
    else:
        print('La carpeta ya existe, ¿la sobreescribimos?')
        response = input()
        if response == 'Sí' or response == 'sí' or response == 'Si' or response == 'si':
            pass
        else:
            num_string = str(int(num_string) + 1)
            general_path = path + f'\\{word_string}-{num_string}'
            os.makedirs(general_path)
        print('Entendido')

    # Recorremos las imágenes recortadas y las almacenamos en la carpeta seleccionada
    for n, image in enumerate(images_list):
        # Comprobamos si la letra es mayúscula o minúscula para etiquetar la imagen
        if word_string[n].isupper() == True:
            word_list_name = str('Capital_' + word_string[n])
        else:  
            word_list_name = str('Lower_' + word_string[n])

        # Almacenamos la imagen en la carpeta seleccionada con la etiqueta adecuada
        image_path = str(general_path + f'\\{word_list_name}-{n}.jpeg')
        
        # Normalizamos la imagen si es necesario
        if normalized == True:
            image = image / np.max(image)    # np.round(image / np.max(image))

        # Mostramos la imagen
        plt.imshow(image, cmap = 'gray')
        plt.show()

        # Guardamos la imagen
        plt.imsave(image_path, image, cmap = 'gray')
    return print('Imágenes almacenadas con éxito')