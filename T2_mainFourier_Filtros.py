import cv2
import numpy as np
import os
import sys

""" =================================================================================
--------TALLER_2 : Banco de filtros orientación - Fourier----------
--------------------Presentado por: Sergio B - Christian Forero.----------------------------
-----------------Alumnos: Maestría en Inteligencia Artificial---------------------
====================================================================================="""

""" Fuente: Código FFT based filtering - Autor: Prof. Julian Quiroga
    python fft_filtering.py <path_to_image> <image_name>
"""

class thetaFilter:

    def __init__(self, img):
        self.image = img

    def  set_theta(self,tetha,D_theta):
        self.tetha=tetha
        self.D_theta=D_theta
        return self.tetha, self.D_theta

    def filtering (self):

        image_gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        image_gray_fft = np.fft.fft2(image_gray)
        image_gray_fft_shift = np.fft.fftshift(image_gray_fft)

        # fft visualization
        image_gray_fft_mag = np.absolute(image_gray_fft_shift)
        image_fft_view = np.log(image_gray_fft_mag + 1)
        image_fft_view = image_fft_view / np.max(image_fft_view)

        # pre-computations
        num_rows, num_cols = (image_gray.shape[0], image_gray.shape[1])
        enum_rows = np.linspace(0, num_rows - 1, num_rows)
        enum_cols = np.linspace(0, num_cols - 1, num_cols)
        col_iter, row_iter = np.meshgrid(enum_cols, enum_rows)
        half_size = num_rows / 2  # here we assume num_rows = num_columns

        # orientation-based filter mask
        orientation_mask = np.zeros_like(image_gray)

        # Verficamos la orientación del píxel
        orientation = 180 * np.arctan2(row_iter - half_size, col_iter - half_size) / np.pi + 180
        idx_orientation_larger = orientation < self.tetha + self.D_theta
        idx_orientation_lesser = orientation > self.tetha - self.D_theta
        idx_orientation_1 = np.bitwise_and(idx_orientation_larger,
                                           idx_orientation_lesser)  # Unimos las orientaciones mediante una operación lógica (AND)
        idx_orientation_larger = orientation < self.tetha + 180 + self.D_theta
        idx_orientation_lesser = orientation > self.tetha + 180 - self.D_theta
        idx_orientation_2 = np.bitwise_and(idx_orientation_larger, idx_orientation_lesser)
        idx_orientation = np.bitwise_or(idx_orientation_1, idx_orientation_2)
        orientation_mask[idx_orientation] = 1
        orientation_mask[int(half_size), int(half_size)] = 1

        # filtering via FFT
        mask = orientation_mask  # can also use high or band pass mask
        fft_filtered = image_gray_fft_shift * mask
        image_filtered = np.fft.ifft2(np.fft.fftshift(fft_filtered))
        image_filtered = np.absolute(image_filtered)
        image_filtered /= np.max(image_filtered)

        return image_filtered, mask

def std_N (image, N, thresh):

    # Calculamos (M1 - M2) vecindarios de la imagen
    image_x = cv2.blur(image, (N, N))
        # Elevamos cada pixel de la imagen al cuadrado
    image_x_2 = cv2.blur(np.power(image, 2 ), (N, N))
        # Calculo desviación estándar
    image_std = np.sqrt(image_x_2 - np.power(image_x, 2))
        #  Normalizamos la imagen
    image_std /= np.max(image_std)
    mask_std = image_std > thresh

    return mask_std

if __name__ == '__main__':

    img = cv2.imread("01_1.tif")
    filtrada0 = thetaFilter(img)
    filtrada0.set_theta(0, 20)
    img_0 = filtrada0.filtering()[0]

    img_45 = cv2.imread("01_1.tif")
    filtrada45 = thetaFilter(img_45)
    filtrada45.set_theta(45, 20)
    img_45 = filtrada45.filtering()[0]
    mask_45= filtrada45.filtering()[1]

    img_90 = cv2.imread("01_1.tif")
    filtrada90 = thetaFilter(img_90)
    filtrada90.set_theta(90, 20)
    img_90 = filtrada90.filtering()[0]

    img_135 = cv2.imread("01_1.tif")
    filtrada135 = thetaFilter(img_135)
    filtrada135.set_theta(135, 20)
    img_135 = filtrada135.filtering()[0]

    ## Método desviación estandar
    # Calculamos el promedio con una ventana de 11
    N = 11
    # Valor Thresh
    thresh = 0.5
    # Definimos máscara con la desviación estándar local
    mask_std_0 = std_N(img_0, N, thresh)
    mask_std_45 = std_N(img_45, N, thresh)
    mask_std_90 = std_N(img_90, N, thresh)
    mask_std_135 = std_N(img_135, N, thresh)

    # Creamos la imagen sintetizada
    resulting_image = mask_std_0.astype(np.float) * img_0 + mask_std_45.astype(
        np.float) * img_45 + mask_std_90.astype(np.float) * img_90 + mask_std_135.astype(np.float) * img_135

    # Hacemos el promedio de las máscaras de la stdr local
    sum_mask = mask_std_0.astype(np.float) + mask_std_45.astype(
        np.float) + mask_std_90.astype(np.float) + mask_std_135.astype(np.float)

    cv2.imshow("Original image", img)
    #cv2.imshow("Filter frequency response", 255 * mask_45)


    #cv2.imshow("45° Std image", 255 * (mask_std_45.astype(np.uint8)))
    cv2.imshow("0 Orientation Filtered image", img_0)
    cv2.imshow("45 Orientation Filtered image", img_45)
    cv2.imshow("90 Orientation Filtered image", img_90)
    cv2.imshow("135 Orientation Filtered image", img_135)

    cv2.imshow("Resulting image", resulting_image)
    cv2.imshow("Sum mask", sum_mask/4)
    cv2.waitKey(0)
