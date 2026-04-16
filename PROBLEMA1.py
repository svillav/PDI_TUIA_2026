#################################################
#                                               #
# Problema 1 - Ecualización local de histograma #
#                                               #
#################################################

import cv2
import numpy as np
import matplotlib.pyplot as plt

def ecualizacion_local_histograma(img, M, N):
    # Centro de la ventana
    centro_M = M // 2
    centro_N = N // 2

    # Borde (padding) a la imagen original
    img_border = cv2.copyMakeBorder(img, centro_M, centro_M, centro_N, centro_N, borderType=cv2.BORDER_REPLICATE)

    # Matriz vacía, del mismo tamaño de la imagen original
    img_salida = np.zeros_like(img)
    
    # Dimensiones de la imagen original
    filas, columnas = img.shape

    # Barrido píxel por píxel
    for i in range(filas):
        for j in range(columnas):
            # Slicing del ROI
            roi = img_border[i : i+M, j : j+N]

            # Ecualización de histograma
            roi_eq = cv2.equalizeHist(roi)        

            # Carga del píxel central ecualizado
            img_salida[i, j] = roi_eq[centro_M, centro_N]

    return img_salida

def main():
    # Carga de imagen con escala de grises
    path = 'Imagen_con_detalles_escondidos.tif'        
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    # Ecualización de imagenes con distintos tamaños de ventana
    img_eq_7 = ecualizacion_local_histograma(img, 7, 7)
    img_eq_17 = ecualizacion_local_histograma(img, 17, 17)
    img_eq_h = ecualizacion_local_histograma(img, 5, 30)
    img_eq_v = ecualizacion_local_histograma(img, 55, 1)
    img_eq_71 = ecualizacion_local_histograma(img, 71, 71)

    # Visualización de resultados
    plt.figure(figsize=(15, 10))

    plt.subplot(231)
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.title('Original')
    plt.colorbar()
    
    plt.subplot(232)
    plt.imshow(img_eq_7, cmap='gray', vmin=0, vmax=255)
    plt.title('7x7')

    plt.subplot(233)
    plt.imshow(img_eq_17, cmap='gray', vmin=0, vmax=255)
    plt.title('17x17')

    plt.subplot(234)
    plt.imshow(img_eq_h, cmap='gray', vmin=0, vmax=255)
    plt.title('Ventana horizontal 5x30')

    plt.subplot(235)
    plt.imshow(img_eq_v, cmap='gray', vmin=0, vmax=255)
    plt.title('Ventana vertical 55x1')

    plt.subplot(236)
    plt.imshow(img_eq_71, cmap='gray', vmin=0, vmax=255)
    plt.title('71x71')

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.2)
    plt.show()

if __name__ == "__main__":
    main()