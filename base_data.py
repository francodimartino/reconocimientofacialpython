import cv2   # Importamos la biblioteca de OpenCV
import os    # Importamos la biblioteca para trabajar con archivos y carpetas del sistema operativo
import imutils   # Importamos la biblioteca para procesamiento de imágenes

# Pedimos al usuario que ingrese el nombre de la persona a la que se le tomarán las fotos
nombrePersona=input('Nombre de la persona: ')

pathDatos= 'data/'   # Definimos la ruta de la carpeta de datos
pathPersona= pathDatos + nombrePersona + '/'   # Definimos la ruta de la carpeta de la persona

# Verificamos si la carpeta de la persona ya existe, si no, la creamos
if not os.path.exists(pathPersona):
    print('Carpeta creada: ', pathPersona)
    os.makedirs(pathPersona)

# Creamos una instancia de la cámara web (0) y definimos el tipo de captura
captura = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Cargamos el clasificador pre-entrenado para la detección de caras
clasificadorCaras= cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

count=0  # Inicializamos el contador de imágenes tomadas

# Iniciamos un ciclo para la captura de imágenes
while True:
    ret, frame =captura.read()  # Leemos un frame desde la cámara

    # Si no se pudo leer correctamente un frame, salimos del ciclo
    if ret == False:
        break

    frame = imutils.resize(frame, width=640)  # Redimensionamos el frame a un ancho de 640 píxeles
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convertimos el frame a escala de grises
    auxFrame = gris.copy()   # Hacemos una copia del frame en escala de grises

    # Detectamos las caras en el frame
    caras= clasificadorCaras.detectMultiScale(gris, 1.3, 4)

    # Recorremos cada cara detectada
    for (x,y,w,h) in caras:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)   # Dibujamos un rectángulo alrededor de la cara
        rostro = auxFrame[y:y+h, x:x+w]  # Extraemos la región de la cara
        rostro = cv2.resize(rostro, (720,720), interpolation=cv2.INTER_CUBIC)   # Redimensionamos la imagen a un tamaño de 720x720
        cv2.imwrite(pathPersona + str(count) + '.jpg', rostro)  # Guardamos la imagen de la cara en la carpeta de la persona
        count = count + 1  # Incrementamos el contador de imágenes

    cv2.imshow('video', frame)  # Mostramos el frame en una ventana llamada 'video'

    k= cv2.waitKey(1)  # Esperamos a que el usuario presione una tecla
    if k == 27 or count >= 300:   # Si el usuario presiona la tecla 'ESC' o se han tomado 300 imágenes, salimos del ciclo
        break

captura.release()  # Liberamos la cámara
cv2.destroyAllWindows()  # Cerramos todas las ventanas abiertas por Open