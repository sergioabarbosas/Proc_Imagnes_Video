import cv2
import numpy as np
import os
import sys
import math


""" =================================================================================
--------TALLER_2 : Banco de filtros utilizando la trasnformada de Fourier----------
--------------------Creado por: Sergio B - Christian F.----------------------------
-----------------Alumnos: Maestría en Inteligencia Artificial---------------------
====================================================================================="""


class thetaFilter:

    def __init__(self, img):
	
        self.image = img

    # Se crea el método para recibir parámetros 𝜃 y Δ𝜃 que definen la respuesta del filtro.
    def  set_theta(self,tetha,D_theta):
        self.tetha=math.radians(tetha)
        self.D_theta=math.radians(D_theta)
        self.theta2= self.tetha-math.radians(180)

	# Método para implementar un filtrado FFT componentes frecuencia orientada
    def filtering(self):

        # Convertimos la imagen BGR a escala gris
        image_gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        image_gray_fft = np.fft.fft2(image_gray)
        image_gray_fft_shift = np.fft.fftshift(image_gray_fft)
        image_gray_fft_mag = np.absolute(image_gray_fft_shift)
        num_rows, num_cols = (image_gray.shape[0], image_gray.shape[1])
        enum_rows = np.linspace(0, num_rows - 1, num_rows)
        enum_cols = np.linspace(0, num_cols - 1, num_cols)
        col_iter, row_iter = np.meshgrid(enum_cols, enum_rows)
        # Calculamos el centro de la imagen
        half_size= num_rows / 2 - 1
        half_size2 = num_cols / 2 - 1

        # band pass filter mask
        band_pass_mask = np.zeros_like(image_gray)

        # Se crea el if para los theta 0°, 45°, 135°, 180°
        if (self.tetha>= math.radians(0) and self.tetha<= math.radians(45)) or (self.tetha>= math.radians(135) and self.tetha<= math.radians(180)) :

            idx_low = np.arctan2((col_iter - half_size+15),(row_iter - half_size)) > self.theta2
            idx_high = np.arctan2((col_iter - half_size+15),(row_iter - half_size)) < self.tetha
            idx_low2 = np.arctan2((col_iter - half_size2-15), (row_iter - half_size)) > self.theta2
            idx_high3 = np.arctan2((col_iter - half_size2-15), (row_iter - half_size)) < self.tetha
            idx_bp = np.bitwise_and(idx_low, idx_high)
            idx_bp2 = np.bitwise_and(idx_low2, idx_high3)
            idx_bp3 = np.bitwise_xor(idx_bp2, idx_bp)

            idx_low2 = np.arctan2((col_iter - half_size + 15), (row_iter - half_size)) > self.theta2+self.D_theta
            idx_high2 = np.arctan2((col_iter - half_size + 15), (row_iter - half_size)) < self.tetha+self.D_theta
            idx_low22 = np.arctan2((col_iter - half_size2 - 15), (row_iter - half_size)) > self.theta2+self.D_theta
            idx_high32 = np.arctan2((col_iter - half_size2 - 15), (row_iter - half_size)) < self.tetha+self.D_theta
            idx_bp2 = np.bitwise_and(idx_low2, idx_high2)
            idx_bp22 = np.bitwise_and(idx_low22, idx_high32)
            idx_bp32 = np.bitwise_xor(idx_bp22, idx_bp2)

            idx_bp33 = np.bitwise_or(idx_bp32, idx_bp3)

            band_pass_mask[idx_bp33] = 1

            mask = band_pass_mask   # can also use high or band pass mask
            fft_filtered = image_gray_fft_shift * mask
            image_filtered = np.fft.ifft2(np.fft.fftshift(fft_filtered))
            image_filtered = np.absolute(image_filtered)
            image_filtered /= np.max(image_filtered)
            # cv2.imshow("Filter frequency response", 255 * mask)
            # cv2.imshow("Filtered image", image_filtered)
            # cv2.waitKey(0)
            return image_filtered, 255 * mask

        # Se crea esta condición para permitir el filtrado de frecuencias.
        if (self.tetha>= math.radians(46) and self.tetha<= math.radians(134))  :

            idx_low = np.arctan2((col_iter - half_size),(row_iter - half_size+15)) > self.theta2
            idx_high = np.arctan2((col_iter - half_size),(row_iter - half_size+15)) < self.tetha
            idx_low2 = np.arctan2((col_iter - half_size2), (row_iter - half_size-15)) > self.theta2
            idx_high3 = np.arctan2((col_iter - half_size2), (row_iter - half_size-15)) < self.tetha
            idx_bp = np.bitwise_and(idx_low, idx_high)
            idx_bp2 = np.bitwise_and(idx_low2, idx_high3)
            idx_bp3 = np.bitwise_xor(idx_bp2, idx_bp)

            idx_low2 = np.arctan2((col_iter - half_size ), (row_iter - half_size+ 15)) > self.theta2+self.D_theta
            idx_high2 = np.arctan2((col_iter - half_size ), (row_iter - half_size+ 15)) < self.tetha+self.D_theta
            idx_low22 = np.arctan2((col_iter - half_size2 ), (row_iter - half_size- 15)) > self.theta2+self.D_theta
            idx_high32 = np.arctan2((col_iter - half_size2 ), (row_iter - half_size- 15)) < self.tetha+self.D_theta
            idx_bp2 = np.bitwise_and(idx_low2, idx_high2)
            idx_bp22 = np.bitwise_and(idx_low22, idx_high32)
            idx_bp32 = np.bitwise_xor(idx_bp22, idx_bp2)

            idx_bp33 = np.bitwise_or(idx_bp32, idx_bp3)

            band_pass_mask[idx_bp33] = 1

            mask = band_pass_mask   # can also use high or band pass mask
            fft_filtered = image_gray_fft_shift * mask
            image_filtered = np.fft.ifft2(np.fft.fftshift(fft_filtered))
            image_filtered = np.absolute(image_filtered)
            image_filtered /= np.max(image_filtered)
            return image_filtered, 255 * mask
            # cv2.imshow("Filter frequency response", 255 * mask)
            # cv2.imshow("Filtered image", image_filtered)
            # cv2.waitKey(0)


if __name__ == '__main__':

    # Se ingresa el path de la imagen (huellas)
    img=cv2.imread("C:\\Users\\FAMILIAR\\Desktop\\sbarbosa\\Universidad_Javeriana\\Proc_imagenes_video\\Images\\huellas\\01\\01_1.tif")

    # Se crea el objeto para el filtro de 0°
    filtrada1=thetaFilter(img)
    filtrada1.set_theta(0, 5)
    imagen1 = filtrada1.filtering()[0]

    # Se crea el objeto para el filtro de 45°
    filtrada2 = thetaFilter(img)
    filtrada2.set_theta(45, 5)
    imagen2 = filtrada2.filtering()[0]

    # Se crea el objeto para el filtro de 90°
    filtrada3 = thetaFilter(img)
    filtrada3.set_theta(90, 5)
    imagen3 = filtrada3.filtering()[0]

    # Se crea el objeto para el filtro de 135°
    filtrada4 = thetaFilter(img)
    filtrada4.set_theta(135, 5)
    imagen4 = filtrada4.filtering()[0]

    # Se pormedian
    final_image = cv2.addWeighted(imagen1, 0.5, imagen2, 0.5, 0.0)
    final_image2 = cv2.addWeighted(imagen3, 0.5, imagen4, 0.5, 0.0)
    final_image3 = cv2.addWeighted(final_image, 0.5, imagen2, 0.5, 0.0)

    # Visualizar las imagenes con filtros
    cv2.imshow("Final image ",final_image)
    cv2.imshow(" 0 grados ", imagen1)
    cv2.imshow(" 45 grados ", imagen2)
    cv2.imshow(" 90 grados ", imagen3)
    cv2.imshow(" 135 grados ", imagen4)
    cv2.waitKey(0)