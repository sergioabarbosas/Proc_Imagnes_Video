import cv2
import numpy as np
import os
import sys

""" =================================================================================
--------TALLER_3 : Interpolación, diezmado y descomposición de imágenes ------------
--------------------Creado por: Sergio B - Christian F.----------------------------
-----------------Alumnos: Maestría en Inteligencia Artificial---------------------
====================================================================================="""

""" Fuente: Código Decimation, interpolation and resizing - Autor: Prof. Julian Quiroga
    python decimation_interpolation.py <path_to_image> <image_name>
"""
class deci_inter:

    def decimation (self, img, D):

        self.image = img
        # Convertir la imagen de BGR a GRIS
        try:
            image_gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        except:
            image_gray = self.image
        image_gray_fft = np.fft.fft2(image_gray)
        image_gray_fft_shift = np.fft.fftshift(image_gray_fft)

        # pre-computations
        num_rows, num_cols = (image_gray.shape[0], image_gray.shape[1])
        enum_rows = np.linspace(0, num_rows - 1, num_rows)
        enum_cols = np.linspace(0, num_cols - 1, num_cols)
        col_iter, row_iter = np.meshgrid(enum_cols, enum_rows)
        half_size = num_rows / 2  # here we assume num_rows = num_columns

        # Utilizando la FFT filtre / suprime frecuencias mayores 1 / D(freq_cut_off=1 / D).
        # low pass filter
        low_pass_mask = np.zeros_like(image_gray)
        freq_cut_off = 1 / D  # it should less than 1
        radius_cut_off = int(freq_cut_off * half_size)
        idx_lp = np.sqrt((col_iter - half_size) ** 2 + (row_iter - half_size) ** 2) < radius_cut_off
        low_pass_mask[idx_lp] = 1

        # filtering via FFT
        mask = low_pass_mask  # can also use high or band pass mask
        fft_filtered = image_gray_fft_shift * mask
        image_filtered = np.fft.ifft2(np.fft.fftshift(fft_filtered))
        image_filtered = np.absolute(image_filtered)
        image_filtered /= np.max(image_filtered)

        # Diezmado
        image_decimated = image_filtered[::D, ::D]

        return image_decimated


    def interpolation (self, img, factor_I):

        self.image = img
        # Convertir la imagen de BGR a GRIS
        try:
            image_gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        except:
            image_gray=self.image
        # insert zeros
        rows, cols = image_gray.shape
        # num_of_zeros = factor I - 1
        # Se toma el tamaño de la imagen original y el factor I (Si el I = 5 ; la imagen es x5)
        image_zeros = np.zeros((factor_I * rows, factor_I * cols), dtype=image_gray.dtype)
        # rebuild with zeros (I-1) between samples
        image_zeros[::factor_I, ::factor_I] = image_gray
        # Tamaño de la ventana
        W = 2 * factor_I + 1

        # filtering low pass Blur Gaussiano
        image_interpolated = cv2.GaussianBlur(image_zeros, (W, W), 2.0) # std campana gauss = 2
        image_interpolated *= factor_I ** 2

        image_gray = image_interpolated
        image_gray_fft = np.fft.fft2(image_gray)
        image_gray_fft_shift = np.fft.fftshift(image_gray_fft)

        # pre-computations
        num_rows, num_cols = (image_gray.shape[0], image_gray.shape[1])
        enum_rows = np.linspace(0, num_rows - 1, num_rows)
        enum_cols = np.linspace(0, num_cols - 1, num_cols)
        col_iter, row_iter = np.meshgrid(enum_cols, enum_rows)
        half_size = num_rows / 2  # here we assume num_rows = num_columns

        # Utilizando la FFT filter / suprime frecuencias mayores 1 / D(freq_cut_off=1 / D).
        # low pass filter
        low_pass_mask = np.zeros_like(image_gray)
        freq_cut_off = 1 / factor_I  # it should less than 1
        radius_cut_off = int(freq_cut_off * half_size)
        idx_lp = np.sqrt((col_iter - half_size) ** 2 + (row_iter - half_size) ** 2) < radius_cut_off
        low_pass_mask[idx_lp] = 1

        # filtering via FFT
        mask = low_pass_mask  # can also use high or band pass mask
        fft_filtered = image_gray_fft_shift * mask
        image_filtered = np.fft.ifft2(np.fft.fftshift(fft_filtered))
        image_filtered = np.absolute(image_filtered)
        image_filtered /= np.max(image_filtered)

        return image_filtered
    #
    def descomposition (self, img, N=2):

        # Recibe imagen de entrada y convierte en gris

        self.image = img
        image_gray_desc = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # Se crean los Kernels
        # (H) Horizontal filter
        kernel_H = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        # (V) Vertical filter
        kernel_V = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        # (D) Diagonal filter
        kernel_D = np.array([[2, -1, -2], [-1, 4, -1], [-2, -1, 2]])
        # (L) Low-pass filter
        kernel_L = np.array([[1/16, 1/8, 1/16], [1/8, 1/4, 1/8], [1/16, 1/8, 1/16]])

        # ----- Descomposición de Orden 1 -------
        # La imagen I es filtrada (convolución) con cada uno de los Kernels I*H, I*V, I*D e I*L
        # cv2.filter2D(src=image, ddepth=-1, kernel=kernel1)
        self.lista_principal = []

        for i in range (N):
            lista=[0 for i in range(4)]

            if i==0:

                image_convolved_IH = cv2.filter2D(image_gray_desc, -1, kernel_H)
                image_convolved_IV = cv2.filter2D(image_gray_desc, -1, kernel_V)
                image_convolved_ID = cv2.filter2D(image_gray_desc, -1, kernel_D)
                self.image_convolved_IL = cv2.filter2D(image_gray_desc, -1, kernel_L)

                lista[0] = self.decimation (image_convolved_IH, 2)
                lista[1] = self.decimation (image_convolved_IV, 2)
                lista[2] = self.decimation (image_convolved_ID, 2)
                lista[3] = self.decimation (self.image_convolved_IL, 2)

                self.image_convolved_IL=lista[3]
                self.lista_principal.append(lista)

            else:
                image_convolved_IH = cv2.filter2D(self.image_convolved_IL, -1, kernel_H)
                image_convolved_IV = cv2.filter2D(self.image_convolved_IL, -1, kernel_V)
                image_convolved_ID = cv2.filter2D(self.image_convolved_IL, -1, kernel_D)
                image_convolved_IL = cv2.filter2D(self.image_convolved_IL, -1, kernel_L)

                lista[0] = self.decimation(image_convolved_IH, 2)
                lista[1] = self.decimation(image_convolved_IV, 2)
                lista[2] = self.decimation(image_convolved_ID, 2)
                lista[3] = self.decimation(image_convolved_IL, 2)

                self.image_convolved_IL = lista[3]
                self.lista_principal.append(lista)

        return self.lista_principal



if __name__ == '__main__':

    # path = sys.argv[1]
    # image_name = sys.argv[2]
    # path_file = os.path.join(path, image_name)
    # Lectura de la imagen
    img = cv2.imread("lena.png")

    # Diezmado
    # Crear objeto para el método diezmado (imagen, factor D > 1)
    #metodo_decimation = deci_inter().decimation (img, 2)
    #cv2.imshow("Decimation image", metodo_decimation)
    #cv2.waitKey(0)

    # Interpolation
    # Ingreso al método el Factor I-1 = num_of_zeros
    #metodo_interpolation = deci_inter().interpolation(img,2)
    #cv2.imshow("Interpolation image", metodo_interpolation)
    #cv2.waitKey(0)

    # Descomposition
    clase=deci_inter()
    metodo_descomposition_H_1 = clase.descomposition(img, 2)[0][0]
    metodo_descomposition_V_1 = clase.descomposition(img, 2)[0][1]
    metodo_descomposition_D_1 = clase.descomposition(img, 2)[0][2]
    metodo_descomposition_L_1 = clase.descomposition(img, 2)[0][3]
    metodo_descomposition_H_2 = clase.descomposition(img, 2)[1][0]
    metodo_descomposition_V_2 = clase.descomposition(img, 2)[1][1]
    metodo_descomposition_D_2 = clase.descomposition(img, 2)[1][2]
    metodo_descomposition_L_2 = clase.descomposition(img, 2)[1][3]

    cv2.imshow("IH_1", metodo_descomposition_H_1)
    cv2.imshow("IV_1", metodo_descomposition_V_1)
    cv2.imshow("ID_1", metodo_descomposition_D_1)
    cv2.imshow("IL_1", metodo_descomposition_L_1)
    cv2.imshow("ILH_1", metodo_descomposition_H_2)
    cv2.imshow("ILV_1", metodo_descomposition_V_2)
    cv2.imshow("ILD_1", metodo_descomposition_D_2)
    cv2.imshow("ILL_1", metodo_descomposition_L_2)

    metodo_interpolation = clase.interpolation(metodo_descomposition_L_2,4)
    cv2.imshow("ILL_INTERPOLATE", metodo_interpolation)
    # cv2.waitKey(0)

    cv2.waitKey(0)


