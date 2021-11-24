"""
============= Examen FINAL : Alumno Sergio Barbosa =====================
"""

import cv2
import os
import sys
import numpy as np

""" 
    Ingresa <path_to_image> <image_name>
"""

points = []

#===== Parte del punto 3 (método click) =======================0
# Definimos método para el click de las coordenadas homogéneas
def click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Únicamente permite realizar dos puntos para realizar la línea
        if len(points)<2:
            points.append((x, y))
        if len(points)==2:
            cv2.line(image_draw, (points[0]), (points[1]), [0, 255, 255], thickness=2)



if __name__ == '__main__':
    path = sys.argv[1]
    image_name = sys.argv[2]
    path_file = os.path.join(path, image_name)
    image = cv2.imread(path_file)

    # #========================================================================
    # #======================== Punto 1 =======================================
    # # Histograma HUE
    # image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # hist_hue = cv2.calcHist([image_hsv], [0], None, [180], [0, 180])
    # # Hallamos el valor max
    # max_val = hist_hue.max()
    # max_pos = int(hist_hue.argmax())
    # # Peak mask
    # lim_inf = (max_pos - 10, 0, 0)
    # lim_sup = (max_pos + 10, 255, 255)
    # image_mask = cv2.inRange(image_hsv, lim_inf, lim_sup)
    # # Calculo pixeles
    # number_white_pix = np.sum(image_mask == 255)  # extracting only white pixels
    # number_black_pix = np.sum(image_mask == 0)  # extracting only black pixels
    # sum_pixels = number_white_pix + number_black_pix
    # porc_white_pix = (number_white_pix / sum_pixels)
    # print("porcentaje de pixeles blancos es (%): ", porc_white_pix * 100 )
    # # Visualización
    # cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("Image", 1280, 720)
    # cv2.imshow("Image", mask)
    # cv2.waitKey(0)
    #
    # #========================================================================
    # #======================== Punto 2 =======================================
    # # Considero las mask del punto 1 igual para el punto 2
    # mask_punto2 = image_mask
    # # Aplico el kernel de 7X7
    # kernel = np.ones((9, 9), np.uint8)
    # mask_punto2 = cv2.morphologyEx(mask_punto2, cv2.MORPH_OPEN, kernel)
    # mask_not = cv2.bitwise_not(mask_punto2)
    #
    # # Encontrar contornos
    # contours, hierarchy = cv2.findContours(mask_not, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # image_draw = image.copy()
    # jugadores = 0
    # for idx, cont in enumerate(contours):
    #     if len(contours[idx]) > 20:
    #         hull = cv2.convexHull(contours[idx])
    #         M = cv2.moments(contours[idx])
    #         cx = int(M['m10'] / M['m00'])
    #         cy = int(M['m01'] / M['m00'])
    #         area = M['m00']
    #         if area < 5000 and area > 500:
    #             x, y, width, height = cv2.boundingRect(contours[idx])
    #             cv2.rectangle(image_draw, (x, y), (x + width, y + height), (0, 0, 255), 2)
    #             # Contador jugadores
    #             jugadores = jugadores + 1
    #
    # print(f"total de jugadores en la cancha: {jugadores}")
    # cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("Image", 1080, 720)
    # cv2.imshow("Image", image_draw)
    # cv2.waitKey(0)

    #========================================================================
    #======================== Punto 3 =======================================

    image_draw = np.copy(image)
    # Creo dos listas vacías para los puntos de las rectas
    points1 = []
    points2 = []
    # Seleccionando con el click los puntos
    cv2.namedWindow("Image")
    cv2.setMouseCallback("Image", click)
    # Contador de puntos
    point_counter = 0
    while True:
        cv2.imshow("Image", image_draw)
        key = cv2.waitKey(1) & 0xFF
        if len(points) == 2:
            points1 = points.copy()
            points = []
            break
        # if len(points) > point_counter:
        #     point_counter = len(points)
        #     cv2.circle(image_draw, (points[-1][0], points[-1][1]), 3, [0, 0, 255], -1)

    # N = min(len(points1)
    # assert N >= 1, 'At least two points are required'

    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Image", 1080, 720)
    cv2.imshow("Image", image_draw)
    cv2.waitKey(0)