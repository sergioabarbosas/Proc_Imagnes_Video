import cv2
import sys
import os
import numpy as np

#from matplotlib import pyplot as plt

# ===================================================================
#           Taller 2 : Propiedades de contornos
# ===================================================================

# python T2_Contornos.py <path_to_image> <image_name>
# C:\Users\FAMILIAR\Desktop\sbarbosa\Universidad_Javeriana\Proc_imagenes_video\Images placa1.png


if __name__ == '__main__':
    # Recibe el path / imagen.png y la lee (placaX)
    path = sys.argv[1]
    image_name = sys.argv[2]
    path_file = os.path.join(path, image_name)
    image = cv2.imread(path_file)
    image_draw = image.copy()

    # convertimos la imagen BGR a HSV --- BGR a YCRCB [YUV: Y(Luminancia) UV(Crominancia)]
    image_gray = cv2.cvtColor(image_draw, cv2.COLOR_BGR2GRAY)
    image_hsv = cv2.cvtColor(image_draw, cv2.COLOR_BGR2HSV)
    image_YCrCb = cv2.cvtColor(image_draw, cv2.COLOR_BGR2YCR_CB)

    # Binarización global (método de Otsu) ---
    ret, Ibw_Cb = cv2.threshold(image_YCrCb[..., 2], 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Calculamos el histograma de la componente Hue
    hist_hue = cv2.calcHist([image_hsv], [0], Ibw_Cb, [180], [0, 180])
    # Visualizamos histograma
    # plt.plot(hist_hue, color='red')
    # plt.xlim([0, 180])
    # plt.show()

    # Hue: valor máximo y posición
    max_val = hist_hue.max()
    max_pos = int(hist_hue.argmax())

    # Definimos límites y máscara
    lim_inf = (max_pos - 10, 0, 0)
    lim_sup = (max_pos + 10, 255, 255)
    mask_plate = cv2.inRange(image_hsv, lim_inf, lim_sup)

    # mask
    ret, Ibw_sat = cv2.threshold(image_hsv[..., 1], 128, 255, cv2.THRESH_BINARY)
    mask = cv2.bitwise_and(mask_plate, Ibw_sat)
    mask_ = np.logical_and(mask_plate, Ibw_sat)

    W = 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * W + 1, 2 * W + 1))
    mask_eroded = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    W = 5
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * W + 1, 2 * W + 1))
    mask_dilated = cv2.morphologyEx(mask_eroded, cv2.MORPH_CLOSE, kernel)

    #***** 2. Visualizar un rectángulo rotado que encierre la placa, sobre la imagen original*********

    # Aplicamos el contorno y la jerarquía (mask_dilated.
    contours, hierarchy = cv2.findContours(mask_dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    color = (255, 255, 0)

    for idx, cont in enumerate(contours):
        if len(contours[idx]) > 10:
            # Encontrar el contorno convexo
            hull = cv2.convexHull(contours[idx])
            # Pintamos el contorno (amarillo)
            cv2.drawContours(image_draw, contours, idx, (0, 255, 255), 2)
            # Pintamos la envoltura convexa (azul)
            cv2.drawContours(image_draw, [hull], 0, (255, 0, 0), 2)
            # Calculando los momentos
            M = cv2.moments(contours[idx])
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            area = M['m00']
            # Método para pintar un rectángulo alineado con los ejes (Contorno rojo)
            x, y, width, height = cv2.boundingRect(contours[idx])
            cv2.rectangle(image_draw, (x, y), (x + width, y + height), (0, 0, 255), 2)

    cv2.imshow("Image placa (pix. blancos)", mask_dilated)
    cv2.imshow("Image", image_draw)
    cv2.waitKey(0)


