# Proyecto de reconocimiento facial

Este proyecto permite el registro de imágenes para crear una base de datos, el entrenamiento de un modelo de reconocimiento facial y la identificación de caras en tiempo real a partir de una cámara.

## Requisitos

- Python 3.7 o superior
- pip

## Instalación

1. Clonar el repositorio en tu ordenador:

    ```
    git clone https://github.com/francodimartino/reconocimientofacialpython.git
    ```

2. Crear un entorno virtual:

    ```
    python -m venv env
    ```

3. Activar el entorno virtual:

    ```
    source env/bin/activate
    ```

4. Instalar las dependencias del proyecto:

    ```
    pip install -r requirements.txt
    ```

## Uso

### Registro de imágenes

1. Ejecutar el archivo `base_data.py`.

2. Ingresar el nombre de la persona a registrar.

3. Posicionarse frente a la cámara y esperar a que se tomen las imagenes.

4. Presionar la tecla `ESC` para guardar las imágenes y salir del programa.

### Entrenamiento del modelo

1. Ejecutar el archivo `entrenamiento.py`.

2. Esperar a que el modelo sea entrenado.

3. Cerrar la ventana de la terminal.

### Identificación de caras

1. Ejecutar el archivo `reconocedor.py`.

2. Posicionarse frente a la cámara.

3. Esperar a que el programa identifique la cara.

4. Si la persona es conocida, se mostrará su nombre en la pantalla. De lo contrario, se mostrará "Desconocido".

5. Presionar la tecla `ESC` para cerrar el programa.

## Créditos

Este proyecto fue creado por [Franco Di Martino](https://github.com/francodimartino) 
