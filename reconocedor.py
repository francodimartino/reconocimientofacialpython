import cv2
import os

# Directorio de los datos
pathDatos = 'data/'

# Lista de personas existentes en el directorio de datos
listaPersonas = os.listdir(pathDatos)
print('Lista de personas: ', listaPersonas)

# Crea el objeto reconocedor de rostros
reconocedor = cv2.face.LBPHFaceRecognizer_create()

# Lee el modelo entrenado
reconocedor.read('modeloRostros.xml')

# Captura de video desde la cámara
captura = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Cargar clasificador de Haar Cascade para detección de caras
clasificadorCaras = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Ciclo infinito para mostrar el video
while True:

    # Lee un frame del video
    ret, frame = captura.read()
    
    # Si no se pudo leer el frame, termina el ciclo
    if ret == False: 
        break
    
    # Convierte el frame a escala de grises
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Copia del frame en escala de grises
    auxFrame = gris.copy()

    # Detección de caras en el frame
    caras = clasificadorCaras.detectMultiScale(gris, 1.3, 4)

    # Ciclo para cada cara detectada
    for (x,y,w,h) in caras:
        
        # Recorta la región del rostro
        rostro = auxFrame[y:y+h, x:x+w]
        
        # Redimensiona la región del rostro a 150x150
        rostro = cv2.resize(rostro, (150,150), interpolation=cv2.INTER_CUBIC)
        
        # Realiza la predicción del rostro en el modelo entrenado
        resultado = reconocedor.predict(rostro)
        print(resultado)

        # Escribe la predicción en el frame
        cv2.putText(frame, '{}'.format(resultado), (x,y-5), 1, 1.3, (255,255,0), 1, cv2.LINE_AA)

        # Si el error de la predicción es menor a 75, se reconoce a la persona
        if resultado[1] < 75:
            # Escribe el nombre de la persona reconocida en el frame
            cv2.putText(frame, '{}'.format(listaPersonas[resultado[0]]), (x,y-25), 2, 1.1, (0,255,0), 1, cv2.LINE_AA)
            # Dibuja un rectángulo alrededor del rostro
            #cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
        else:
            # Escribe 'Desconocido' en el frame
            cv2.putText(frame, 'Desconocido', (x,y-20), 2, 0.8, (0,0,255), 1, cv2.LINE_AA)
            # Dibuja un rectángulo alrededor del rostro
            #cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 2)

    # Muestra el frame en una ventana
    cv2.imshow('frame', frame)

    # Espera la tecla ESC para salir del programa
   

    k= cv2.waitKey(1)
    if k == 27:
        break
captura.release()
cv2.destroyAllWindows()
