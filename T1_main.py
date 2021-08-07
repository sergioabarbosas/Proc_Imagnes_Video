import cv2
import numpy as np


""" --------TALLER_1 : Leer y mostrar una imagen con OpenCV----------
--------------Creado por: Sergio A. Barbosa--------------------------
------------Alumno: Maestría en Inteligencia Artificial-----------"""

#path TEST = C:\Users\FAMILIAR\Desktop\sbarbosa\Universidad_Javeriana\Proc_imagenes_video\Images\cat.png

# Defino la clase BasicColor
class BasicColor():

    # Constructor para el ruta de la imagen y almacenamiento
    def __init__ (self, path):
        self.image = cv2.imread(path)
        assert self.image is not None, "There is no image at {}".format(path)

    # Método para visualizar número de píxeles (MP) y número de canales
    def displayProperties (self):
        height = self.image.shape[0]
        width = self.image.shape[1]
        channels = self.image.shape[2]
        # Imprimo los # de píxeles (h*w*c) y canales (columna 3 (en python col2)
        print("Numero de pixeles: {}, Canales: {}".format(width*height*channels, channels))

    # Método para retornar versión binaria (método de Otsu) de la imagen
    def makeBW (self):
        # convierto la imagen a gris
        image_gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY) # 0-255
        # ____Aplico el método Otsu____
        ret, Ibw_otsu = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        #retorno la imagen
        return Ibw_otsu

    # Método para generar version colorizada de la imagen
    def colorize(self):
        # Transformando imagen BGR a HSV
        image_hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        # separo la imagen por componentes, transformando h y dejando s y v intactas
        h, s, v = cv2.split(image_hsv)
        # Pido al usuario ingresar valor de h para modificarla
        valor_h = int(input("Ingrese valor h (0-179): "))
        # Transformando los valores de h con el dato ingresado por el usuario
        h = valor_h * np.ones_like(s)  # 0 - 179

        # Reconstruyendo la imagen
        image_hue = cv2.merge((h, s, v))
        # Transformo de HSV a BGR
        image_hue_bgr = cv2.cvtColor(image_hue, cv2.COLOR_HSV2BGR)

        # Concatenando las dos imágenes para visualizar
        concat_horizontal = cv2.hconcat([image_hue_bgr, self.image])

        return concat_horizontal


if __name__ == '__main__':
    # Solicito ingreso de path al usuario
    path=input("introduzca sin comillas la ruta con que que desea trabajar: ")
    # Creamo el objeto imagen para la clase BasicColor
    imagen=BasicColor(path)
    # Retorno el método displayProperties
    imagen.displayProperties()
    # Visualizo la imagen umbralizada con Otsu, invocando el método makeBW
    cv2.imshow("Imagen con el metodo OTSU",imagen.makeBW())
    #Creo Image_colorizada para llamar a la imagen_hue_bgr + image del método colorize
    Imagen_colorizada= imagen.colorize()
    # Visualizo las imágenes
    cv2.imshow("Imagen colorizada VS Imagen real", Imagen_colorizada)
    cv2.waitKey(0)
