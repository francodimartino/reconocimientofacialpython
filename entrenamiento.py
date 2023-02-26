import cv2
import os
import numpy as np

pathDatos= 'data/'
listaPersonas= os.listdir(pathDatos)
print('Lista de personas: ', listaPersonas)

etiquetas= []
rostrosData= []
count=0

for nameDir in listaPersonas:
    personaPath= pathDatos + nameDir
    print('Leyendo las imágenes')
    for fileName in os.listdir(personaPath):
        print('Rostros: ', nameDir + '/' + fileName)
        etiquetas.append(count)
        rostrosData.append(cv2.imread(personaPath + '/' + fileName, 0))
        imagen= cv2.imread(personaPath + '/' + fileName, 0)

    count = count + 1
    print(count)

reconocedor = cv2.face.LBPHFaceRecognizer_create()

print('Entrenando...')
reconocedor.train(rostrosData, np.array(etiquetas))

reconocedor.write('modeloRostros.xml')

print('Modelo almacenado...')




