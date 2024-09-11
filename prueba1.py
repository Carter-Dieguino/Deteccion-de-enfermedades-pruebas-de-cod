# Se me paso enviarte el código Javi
# Ya lleva las imagenes jaja

import cv2
import numpy as np

def mostrar_instrucciones():
    # Crear una imagen en blanco
    instrucciones_img = np.zeros((400, 800, 3), dtype=np.uint8)
    cv2.putText(instrucciones_img, 'Instrucciones:', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(instrucciones_img, 'Ajusta los sliders para procesar la imagen.', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    cv2.putText(instrucciones_img, 'Presiona "s" para guardar la imagen procesada.', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    cv2.putText(instrucciones_img, 'Presiona "q" para avanzar a la siguiente imagen.', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    cv2.putText(instrucciones_img, 'Presiona cualquier tecla para empezar.', (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    
    # Mostrar la imagen con las instrucciones
    cv2.imshow('Instrucciones', instrucciones_img)
    cv2.waitKey(0)
    cv2.destroyWindow('Instrucciones')

def actualizar_binarizacion(img_gris):
    # Leer el valor del blur desde el slider
    blur_value = cv2.getTrackbarPos('Blur', 'Controles')
    # Aseguramos que el valor de blur sea impar y mayor que 1
    if blur_value > 0:
        blur_value = max(1, blur_value)
        blur_value = blur_value + 1 if blur_value % 2 == 0 else blur_value
        img_gris = cv2.medianBlur(img_gris, blur_value)

    # Leer el valor máximo desde el slider
    valor_maximo = cv2.getTrackbarPos('Valor Maximo', 'Controles')

    # Obtenemos los valores actuales de los sliders
    metodo_adaptativo = cv2.ADAPTIVE_THRESH_MEAN_C if cv2.getTrackbarPos('Método adaptativo', 'Controles') == 0 else cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    tipo_umbral = cv2.THRESH_BINARY if cv2.getTrackbarPos('Tipo de umbral', 'Controles') == 0 else cv2.THRESH_BINARY_INV
    tamano_bloque = cv2.getTrackbarPos('Tamaño del bloque', 'Controles') * 2 + 1
    constante = cv2.getTrackbarPos('Constante', 'Controles')
    
    # Aplicar umbral adaptativo
    imagen_binarizada = cv2.adaptiveThreshold(
        img_gris, valor_maximo, metodo_adaptativo, tipo_umbral, tamano_bloque, constante
    )
    
    # Mostrar la imagen binarizada
    cv2.imshow('Imagen binarizada', imagen_binarizada)
    return imagen_binarizada

# Mostrar instrucciones antes de comenzar el programa
mostrar_instrucciones()

# Crear una ventana para los controles y sliders
cv2.namedWindow('Controles')
cv2.createTrackbar('Método adaptativo', 'Controles', 0, 1, lambda x: None) # 0 = inicio, 1 = fin
cv2.createTrackbar('Tipo de umbral', 'Controles', 0, 1, lambda x: None)
cv2.createTrackbar('Tamaño del bloque', 'Controles', 7, 30, lambda x: None)
cv2.createTrackbar('Constante', 'Controles', 0, 20, lambda x: None)
cv2.createTrackbar('Blur', 'Controles', 0, 20, lambda x: None)
cv2.createTrackbar('Valor Maximo', 'Controles', 255, 255, lambda x: None)

# Iteramos desde 1 hasta 13
for i in range(1, 13):  # El 13 es exclusivo, por lo que el rango será de 1 a 13
    # Formamos el nombre del archivo dinámicamente
    filename = f"photo{i}.jpg"
    
    # Leemos la imagen
    imagen = cv2.imread(filename)
    
    # Verificamos que la imagen se haya cargado correctamente
    if imagen is not None:
        # Convertimos la imagen a escala de grises
        img_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        
        while True:
            imagen_binarizada = actualizar_binarizacion(img_gris)
            key = cv2.waitKey(1) & 0xFF

            # Si se presiona 's', guardar la imagen
            if key == ord('s'):
                tipo_label = 'bin' if cv2.getTrackbarPos('Tipo de umbral', 'Controles') == 0 else 'bininv'
                save_filename = f"images/imagenprocesada_{i}_{tipo_label}.jpg"
                cv2.imwrite(save_filename, imagen_binarizada)
                print(f"Imagen guardada como {save_filename}")
            
            # Si se presiona 'q', salir del bucle
            elif key == ord('q'):
                break

    else:
        print(f"No se pudo cargar la imagen {filename}")

cv2.destroyAllWindows()  # Destruimos todas las ventanas abiertas
