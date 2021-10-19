import cv2
from enum import Enum
import numpy as np
import sys
import os
import glob


def nueva(img,imgw):
    h, w, z = img.shape
    img2 = np.zeros((h, w, 3), dtype="uint8")
    for i in range(w):
        for j in range(h):
            if np.all((imgw[j, i] == 0)):
                img2[j, i] = img[h-j-1, w-i-1]

    ancho = img2.shape[1]  # columnas
    alto = img2.shape[0]  # filas
    M = cv2.getRotationMatrix2D((ancho // 2, alto // 2),-180, 1)
    imageOut = cv2.warpAffine(img2, M, (ancho, alto))

    gray = cv2.cvtColor(imageOut, cv2.COLOR_BGR2GRAY)
    exis= int(cv2.countNonZero(gray)/imageOut.shape[0])

    imageOut = imageOut[0:h, 0:exis]

    return imageOut, exis

class Methods(Enum):
    SIFT = 1
    ORB = 2

if __name__ == '__main__':

    # Ingrese el path de la carpeta de imágenes (image_1, image_2, image_3)
    path = sys.argv[1]
    # Ingrese el dominio de las imágenes
    path_file = path +"/*.jpg"
    path = glob.glob(path_file)


    # Lista para guardar imagenes del folder dado
    cv_img = []

    for img in path:
        n = cv2.imread(img)
        cv_img.append(n)
        print(img)

    # Cálculo de cantidad de imágenes
    count=len(cv_img)
    print("Cantidad de imagenes: ", count)
    NIM = int(input("Numero de la imagen que desea desea usar como referencia: "))
    assert NIM <= count, 'Imagen de referencia fuera de los limites'


    image_1 = cv_img[0]
    image_2 = cv_img[1]
    image_3 = cv_img[2]

    METODO = int(input("Escriba 1 si desea utilizar el metodo SIFT o 2 si desea emplear el metodo ORB: "))
    if METODO ==1:
        method = Methods.SIFT
    else:
        method = Methods.ORB

    def homografia(image_1,image_2):

        image_gray_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
        image_draw_1 = np.copy(image_1)
        image_gray_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)
        image_draw_2 = np.copy(image_2)

        if method == Methods.SIFT:
            sift = cv2.SIFT_create(nfeatures=50)   # shift invariant feature transform
            keypoints_1, descriptors_1 = sift.detectAndCompute(image_gray_1, None)
            keypoints_2, descriptors_2 = sift.detectAndCompute(image_gray_2, None)
        else:
            orb = cv2.SIFT_create(nfeatures=50)     # oriented FAST and Rotated BRIEF
            keypoints_1, descriptors_1 = orb.detectAndCompute(image_gray_1, None)
            keypoints_2, descriptors_2 = orb.detectAndCompute(image_gray_2, None)

        image_draw_1 = cv2.drawKeypoints(image_gray_1, keypoints_1, None)
        image_draw_2 = cv2.drawKeypoints(image_gray_2, keypoints_2, None)

        # Interest points matching
        bf = cv2.BFMatcher(cv2.NORM_L2)
        matches = bf.knnMatch(descriptors_1, descriptors_2, k=1)
        image_matching = cv2.drawMatchesKnn(image_1, keypoints_1, image_2, keypoints_2, matches, None)

        # Retrieve matched points
        points_1 = []
        points_2 = []
        for idx, match in enumerate(matches):
            idx2 = match[0].trainIdx
            points_1.append(keypoints_1[idx].pt)
            points_2.append(keypoints_2[idx2].pt)

        # Compute homography and warp image_1
        H, _ = cv2.findHomography(np.array(points_1), np.array(points_2), method=cv2.RANSAC)
        image_warped = cv2.warpPerspective(image_1, H, (image_1.shape[1], image_1.shape[0]))

        return image_warped

    for i in range(len(cv_img)-1):

        if i==0:

            image_warped= homografia(cv_img[i],cv_img[i+1])
            img_out=cv2.resize(nueva(cv_img[i],image_warped)[0],(nueva(cv_img[i],image_warped)[1],cv_img[i+1].shape[0]))
            img_out2=cv2.hconcat([img_out,cv_img[i+1]])
        else:
            image_warped_2 = homografia(img_out2, cv_img[i+1])
            img_out_2 = cv2.resize(nueva(img_out2, image_warped_2)[0], (nueva(img_out2, image_warped_2)[1], cv_img[i+1].shape[0]))
            img_out2 = cv2.hconcat([img_out_2, cv_img[i+1]])

    cv2.imshow("Image", img_out2)
    cv2.waitKey(0)
