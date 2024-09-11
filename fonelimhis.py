import cv2
import numpy as np
from matplotlib import pyplot as plt

def eliminar_fondo_borroso(img):
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bordes = cv2.Canny(gris, 100, 200)
    kernel = np.ones((5, 5), np.uint8)
    dilatacion = cv2.dilate(bordes, kernel, iterations=1)
    mascara = cv2.threshold(dilatacion, 25, 255, cv2.THRESH_BINARY)[1]
    return cv2.bitwise_and(img, img, mask=mascara)

def deteccion_picos(hist, umbral=0.05):
    picos = []
    valor_max = np.max(hist) * umbral
    for i in range(1, len(hist) - 1):
        if hist[i] > hist[i - 1] and hist[i] > hist[i + 1] and hist[i] > valor_max:
            picos.append(i)
    return picos

def intensidad_promedio(hist):
    total_pixeles = np.sum(hist)
    suma_intensidad = np.sum(hist * np.arange(256))
    return suma_intensidad / total_pixeles

def clasificar_enfermedad(intensidad):
    enfermedades = {
        65.80: "Mancha Negra",
        16.62: "Tizón",
        76.15: "Oidio",
        17.31: "Cancro",
        49.05: "Otras Manchas de Hojas",
        83.35: "Mildiu"
    }
    mas_cercana = min(enfermedades, key=lambda x: abs(x - intensidad))
    return enfermedades[mas_cercana], abs(mas_cercana - intensidad)

def detectar_contornos(img):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    inferior_oscuro = np.array([0, 0, 0])
    superior_oscuro = np.array([180, 255, 50])
    mascara_oscura = cv2.inRange(img_hsv, inferior_oscuro, superior_oscuro)
    contornos, _ = cv2.findContours(mascara_oscura, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contornos, mascara_oscura

def obtener_tratamiento(enfermedad):
    recomendaciones = {
        "Mancha Negra": [
            "Plantar en suelo con buen drenaje.",
            "Proporcionar fertilizante orgánico regularmente.",
            "Eliminar hojas muertas y ramas infectadas.",
            "Desinfectar las podadoras con desinfectante doméstico.",
            "Evitar mojar las hojas, aplicar agua a las raíces.\n"
        ],
        "Tizón": [
            "Cultivar variedades tempranas.",
            "Plantar variedades resistentes como Sarpo Mira y Sarpo Axona.",
            "Destruir partes infectadas por tizón.",
            "Mantener el área limpia de escombros.\n"
        ],
        "Oidio": [
            "Inspeccionar plantas antes de comprarlas.",
            "Eliminar escombros infectados y evitar compostarlos.",
            "Espaciar plantas para aumentar la circulación de aire.",
            "Evitar áreas sombreadas.\n"
        ],
        "Cancro": [
            "Eliminar partes enfermas en clima seco.",
            "Cultivar variedades resistentes.",
            "Evitar exceso de riego y hacinamiento.",
            "Envolver árboles jóvenes para prevenir quemaduras solares.\n"
        ],
        "Otras Manchas de Hojas": [
            "Seguir los consejos para controlar la Mancha Negra.",
            "Eliminar escombros infectados para evitar la propagación.\n"
        ],
        "Mildiu": [
            "Mantener las hojas secas.",
            "Limpiar alrededor de las plantas en otoño.",
            "Usar fungicidas que controlen oidio y mildiu.\n"
        ]
    }
    return recomendaciones.get(enfermedad, ["No hay tratamiento específico disponible.\n"])

def procesar_imagen(ruta_img):
    img = cv2.imread(ruta_img)
    if img is None:
        print(f"Error: Imagen {ruta_img} no encontrada")
        return

    img_procesada = eliminar_fondo_borroso(img)
    enfermedad = graficar_histograma_picos(img_procesada)

    contornos, _ = detectar_contornos(img_procesada)
    img_contornos = cv2.drawContours(img_procesada.copy(), contornos, -1, (0, 255, 0), 2)

    tratamiento = obtener_tratamiento(enfermedad)

    plt.figure(figsize=(10, 7))
    plt.imshow(cv2.cvtColor(img_contornos, cv2.COLOR_BGR2RGB))
    plt.title(f"Imagen Procesada: {enfermedad}")
    plt.axis("off")
    plt.show()

    print(f"Recomendaciones para {enfermedad}:")
    for consejo in tratamiento:
        print(f"- {consejo}")

def graficar_histograma_picos(img):
    if img is None:
        print("No hay imagen para procesar")
        return

    img_gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    hist = cv2.calcHist([img_gris], [0], None, [256], [0, 256]).flatten()
    picos = deteccion_picos(hist)
    intensidad_prom = intensidad_promedio(hist)
    enfermedad, desviacion = clasificar_enfermedad(intensidad_prom)
    print(f"Intensidad Promedio: {intensidad_prom:.2f}")
    print(f"Enfermedad: {enfermedad} (Desviación: {desviacion:.2f})")
    return enfermedad

for i in range(1, 7):
    ruta_img = f'photo{i}.jpg'
    print(f"Procesando {ruta_img}")
    procesar_imagen(ruta_img)
