import cv2
import numpy as np
import os
import sys
from hough import Hough
from collections import defaultdict

""" =================================================================================
----------TALLER No 4 : Generador de cuadriláteros y detector de esquinas -----------
------------------- Creado por: Sergio Barbosa - Christian Forero -------------------
-------------------- Alumnos Maestría en Inteligencia Artificial---------------------
====================================================================================="""

""" Fuente: Código Hough.py y find_lines.py - Autor: Prof. Julian Quiroga"""


class quadrilateral:

    def __init__(self,N):

        self.N=N

        # En caso de recibir un tamaño de la imagen (valor_N) par
        if self.N % 2 == 0:
            print("Tamaño de la imagen: ", self.N)
        # En caso de recibir un valor N impar: generar ***error****
        else:
            print ("Error en el valor de N: ", self.N, "impar")


    def generate (self):

        image = np.zeros((self.N,self.N,3), np.uint8)
        image[:] = (255, 255, 0)

        point1x= np.random.randint(0, self.N/2)
        point1y = np.random.randint(0, self.N / 2)

        point2x = np.random.randint(self.N / 2, self.N)
        point2y = np.random.randint(0, self.N / 2)

        point3x = np.random.randint(0, self.N / 2)
        point3y = np.random.randint(self.N / 2, self.N)

        point4x = np.random.randint(self.N / 2, self.N)
        point4y = np.random.randint(self.N / 2, self.N)


        pts = np.array([[point1x, point1y], [point2x, point2y], [point4x, point4y],[point3x, point3y] ], np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(image, [pts], True, (255, 0, 255))

        return image #imagen RGB de un polígono

    def interseccion(self,line1, line2):

        xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
        ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

        def det(a, b):
            return a[0] * b[1] - a[1] * b[0]

        div = det(xdiff, ydiff)
        if div == 0:
            raise Exception('lines do not intersect')

        d = (det(*line1), det(*line2))
        x = det(d, xdiff) / div
        y = det(d, ydiff) / div
        return x, y


    def DetectCorners (self):

        method = 1
        high_thresh = 600
        img=image.generate()
        bw_edges = cv2.Canny(img, high_thresh * 0.3, high_thresh, L2gradient=True)

        hough = Hough(bw_edges)
        accumulator = hough.standard_transform()

        # establezco el umbral (votos) de la recta (50 pixeles)
        acc_thresh = 50
        # Numero de rectas
        N_peaks = 4
        # ventana para encontrar picos (thetas, rho)
        nhood = [35, 15]
        peaks = hough.find_peaks(accumulator, nhood, acc_thresh, N_peaks)

        _, cols = img.shape[:2]
        # Creamos copia de la imagen
        image_draw = np.copy(img)
        print(peaks)

        lista=list()


        for peak in peaks:
            #print(peak)
            # Recuperamos valores de rho y theta
            rho = peak[0]
            theta_ = hough.theta[peak[1]]
            ##print(theta_)

            # Convertimos a radianes
            theta_pi = np.pi * theta_ / 180
            theta_ = theta_ - 180
            # Dibujamos la recta a partir de rho y theta
            a = np.cos(theta_pi)
            b = np.sin(theta_pi)
            x0 = a * rho + hough.center_x
            y0 = b * rho + hough.center_y

            c = -rho
            x1 = int(round(x0 + cols * (-b)))
            y1 = int(round(y0 + cols * a))
            x2 = int(round(x0 - cols * (-b)))
            y2 = int(round(y0 - cols * a))
            #print(self.interseccion((x1, y1), (x2, y2)))
            lista.append(((x1,y1),(x2,y2)))

            cv2.circle(image_draw, (int(x0), int(y0)), 10, (255, 255, 255), 2)

            # Para pintar de diferentes colores
            if np.abs(theta_) < 80:
                image_draw = cv2.line(image_draw, (x1, y1), (x2, y2), [0, 255, 255], thickness=2)
            elif np.abs(theta_) > 100:
                image_draw = cv2.line(image_draw, (x1, y1), (x2, y2), [255, 0, 255], thickness=2)
            else:
                if theta_ > 0:
                    image_draw = cv2.line(image_draw, (x1, y1), (x2, y2), [0, 255, 0], thickness=2)
                else:
                    image_draw = cv2.line(image_draw, (x1, y1), (x2, y2), [0, 0, 255], thickness=2)

        return bw_edges, image_draw

if __name__ == '__main__':

    image = quadrilateral(200)
    imagen=image.DetectCorners()[0]
    imagen2 = image.DetectCorners()[1]

    #cv2.imshow(" ", imagen)
    cv2.imshow("", imagen2)
    cv2.waitKey(0)