
""" =================================================================================
----------TALLER No 5 : Transformaciones espaciales - Homografía --------------------
------------------- Creado por: Sergio Barbosa - Christian Forero -------------------
-------------------- Alumnos Maestría en Inteligencia Artificial---------------------
====================================================================================="""

""" Fuente: Código Homography.py - Autor: Prof. Julian Quiroga"""

#Importación de librerias
import cv2
import sys
import os
import numpy as np
import glob

#Función para guardar Clicks

def click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))

# Función para promediar imagenes

def promedio(img, img2):

    h, w, z = img.shape

    img2 = cv2.resize(img2, (h, w))

    for i in range(w):
        for j in range(h):
            if np.all((img[j, i] == 0)) and np.any((img2[j, i] != 0)):
                img[j, i] = img2[j, i]
            elif np.all((img2[j, i] == 0)) and np.any((img[j, i] != 0)):
                img[j, i] = img[j, i]
            else:
                img[j, i][0] = int((int(img[j, i][0]) + int(img2[j, i][0])) / 2)
                img[j, i][1] = int((int(img[j, i][1]) + int(img2[j, i][1])) / 2)
                img[j, i][2] = int((int(img[j, i][2]) + int(img2[j, i][2])) / 2)
    return  img


if __name__ == '__main__':
    
    # lectura de archivos en carpeta especificada como argumento de la función glob
    # Ingrese el path de la carpeta donde se encuentran las imágenes (image_1, image_2,image_3) e incluya el *. con el tipo de imagen.
    # Ejemplo: "C:/Users/Christian Forero/Desktop/imagenes/*.jpeg"
    path = glob.glob("C:/Users/Christian Forero/Desktop/imagenes/*.jpeg")
    count= len(path)
    
    #Calcúlo de cantidad de imágenes
    print("Cantidad de imagenes: ",count)
    NIM= int(input("Numero de la imagen que desea desea usar como referencia"))
    assert NIM <= count, 'Imagén de referencia fuera de los limites'

    # Lista para guardar imagenes del folder dado
    cv_img = []
    # lista para guardar imagenes de salida de homografias
    cv_img2= []
    # lista para almacenar Homografía
    Homografias = []
    
    for img in path:
        n = cv2.imread(img)
        cv_img.append(n)
    
    #Se muestran las imagenes por pares sucesivos y se determinan los puntos para la homografia
    for i in range(count-1):
        points = []
        points1 = []
        points2 = []

        resized = cv2.resize(cv_img[i], (450,450), interpolation = cv2.INTER_AREA)
        resized2 = cv2.resize(cv_img[i+1], (450,450), interpolation = cv2.INTER_AREA)

        image  = resized
        image2 = resized2

        im_v = cv2.hconcat([image, image2])
        im_v_draw = np.copy(im_v)

        point_counter = 0



        cv2.namedWindow("Image")
        cv2.setMouseCallback("Image", click)
        k=0
        im=[]
        im2=[]

        while True:
            cv2.imshow("Image", im_v_draw)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("x"):
                points2=points.copy()
                im = [points[i] for i in im]
                im2 = [points[j] for j in im2]
                break
            if len(points) > point_counter:
                if len(points)%2!=0:

                    point_counter = len(points)
                    im.append(k)
                    cv2.circle(im_v_draw, (points[-1][0], points[-1][1]), 3, [0, 0, 255], -1)
                    k = k + 1

                else:

                    point_counter = len(points)
                    im2.append(k)
                    cv2.circle(im_v_draw, (points[-1][0], points[-1][1]), 3, [255, 0, 0], -1)
                    k = k + 1

        h, w, c = im_v_draw.shape
        w=w/2
        points1=im
        points2=im2

        N = min(len(points), len(points2))
        assert N >= 4, 'At least four points are required'

        print(points2[:N])
        b=0

        #Se corrige el desplazamiento generado por la concatenación horizontal de las imagenes
        for i in points2:

            a=(i[0]-w,i[1])
            points2[b]=a
            b=b+1


        #Puntos para el cálculo de la homografía
        pts1 = np.array(points1[:N])
        pts2 = np.array(points2[:N])


        if False:
            H, _ = cv2.findHomography(pts1, pts2, method=0)
            Homografias.append(H)
        else:
            H, _ = cv2.findHomography(pts1, pts2, method=cv2.RANSAC)
            Homografias.append(H)


    vh=0
    i=0
    j=0

    #Cálculo de la perspectiva homografica de acuerdo con la posición relativa entre una imagén x y la imagén de referencia
    while i < len(cv_img):

        if j < NIM-1:

            resized = cv2.resize(cv_img[i], (450, 450), interpolation=cv2.INTER_AREA)
            image_warped = cv2.warpPerspective(resized, Homografias[vh], (resized.shape[1], resized.shape[0]))
            i=i+1
            print('i:', i)
            vh=vh+1
            cv_img2.append(image_warped)

        elif j==NIM-1:
            i=i+1

        elif j> NIM-1:
            print('i:',i)
            resized = cv2.resize(cv_img[i], (450, 450), interpolation=cv2.INTER_AREA)
            image_warped = cv2.warpPerspective(resized, Homografias[vh], (resized.shape[1], resized.shape[0]))
            print(i)
            b = np.linalg.inv(Homografias[vh])

            print(vh)
            image_warped = cv2.warpPerspective(resized,b , (resized.shape[1], resized.shape[0]))
            i = i + 1
            cv_img2.append(image_warped)
            vh = vh + 1
        j=j+1

# Obtención de stitching a partir de 3 imagenes
# ---------------------------------------------------------------------------------------

    img=  cv_img2[0]
    img2= cv2.resize(cv_img[1], (450, 450), interpolation=cv2.INTER_AREA)

    img= promedio(img, img2)
    img_1=  cv_img2[1]
    img = promedio(img, img_1)

    cv2.imshow("promedio", img)
    cv2.waitKey(0)
