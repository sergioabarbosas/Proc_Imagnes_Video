"""
Demonstration of the GazeTracking library.
Check the README.md for complete documentation.
"""

import cv2
import pandas as pd

from gaze_tracking import GazeTracking

gaze = GazeTracking()
webcam = cv2.VideoCapture(0)

# Ingrese el path del archivo datos.csv
df= pd.read_csv(r"C:\Users\FAMILIAR\Desktop\sbarbosa\Universidad_Javeriana\2_Proces_Imag_video_JulianQuiroga\
Proyecto\Nivel_atencion\datos.csv")

lista=[]
lista2=[]
#while i<=200:
while True:

    # We get a new frame from the webcam
    _, frame = webcam.read()

    # We send this frame to GazeTracking to analyze it
    gaze.refresh(frame)

    frame = gaze.annotated_frame()
    text = ""

    if gaze.is_blinking():
        text = "Blinking"
        valor= 3.8
    elif gaze.is_right():
        text = "Looking right"
        valor= 0.2
    elif gaze.is_left():
        text = "Looking left"
        valor= 0.8
    elif gaze.is_center():
        text = "Looking center"
        valor=0.5


    left_pupil = gaze.pupil_left_coords()
    right_pupil = gaze.pupil_right_coords()
    #cv2.putText(frame, "Left pupil:  " + str(left_pupil), (20, 380), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 1)
    #cv2.putText(frame, "Right pupil: " + str(right_pupil), (20,440), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 1)

    lista.append(valor)
    if len(lista)>4:
        lista2=lista[-5:]
        media=sum(lista2) / len(lista2)
        if media >0.45 and media <0.75:
            text="Excelente"
        elif media<0.45 or  (media >0.75 and media <1):
            text="Regular"
        else:
            text="Deficiente"

        cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 1)


    #
    # df = df.append({
    #         "Lef": left_pupil,
    #         "Right": right_pupil,
    #         "Label": text
    #
    # }, ignore_index=True)
    # i=i+1

    cv2.imshow("Demo", frame)

    if cv2.waitKey(1) == 27:
        break

df.to_csv(r'C:\Users\Christian Forero\Desktop\Procesamiento de imagenes\datos.csv')
webcam.release()
cv2.destroyAllWindows()