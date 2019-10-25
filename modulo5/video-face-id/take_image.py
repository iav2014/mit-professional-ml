# from graphics import *
import numpy as np
import cv2
# import tkinter as tk
# import tkinter.simpledialog as tksd
import os
from image_functions import *

WEBCAM = 0
DEFAULT_THRESHOLD = 0.5
PATH_TO_DATABASE = 'images'


def capture(model, database, threshold=DEFAULT_THRESHOLD):
  cap = cv2.VideoCapture(WEBCAM)
  
  first_time = True
  while (True):
    if first_time == True:
      tm = 0
      text_info = ""
      first_time = False
    # Capture frame-by-frame
    # Capture frame-by-frameord("T
    ret, frame = cap.read()
    
    identificate, image_face, no_identificate_image = faceID(frame, model, database, threshold=threshold, debug=False)
    
    # Pintamos los recuadros de unknown
    for (img, x, y, w, h) in no_identificate_image:
      frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
      frame = cv2.putText(frame, "Unknown", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    for i in range(len(identificate)):
      id_ident = identificate[i]
      (img, x, y, w, h) = image_face[i]
      frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
      frame = cv2.putText(frame, id_ident, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    if tm > 0:
      tm -= 1
      frame = cv2.putText(frame, text_info, (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    foto = frame  # cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    # Display the resulting frame
    # newimg = cv2.resize(foto, (int(640), int(400)))
    cv2.imshow('video face recognition', foto)
    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('l'):  # load database
      print("l")
      database = dict()
      for curr_image in next(os.walk(PATH_TO_DATABASE))[2]:
        if curr_image != '.DS_Store':  # filter mac empty file...
          name = curr_image.split(".")[0]
          imagen = cv2.imread(os.path.join(PATH_TO_DATABASE, curr_image), 1)
          database[name] = getFeaturesRAW(imagen, model)
    #    if key_pressed == ord('s'):#save database?
    #        print("s")
    if key_pressed == ord('a'):  # aumentar threshold
      print("a")
      threshold += 0.05
      if threshold > 1.0:
        threshold = 1.0
      text_info = "Threshold up: " + str(threshold)
      tm = 150
    if key_pressed == ord('z'):  # reducir threshold ¿cómo mostramos por pantalla durante x secs?
      print("r")
      threshold -= 0.05
      if threshold < 0.1:
        threshold = 0.1
      text_info = "Threshold down: " + str(threshold)
      tm = 150
      # ponemos un mensaje y un timeout para que se vaya (que son x vueltas)
    if key_pressed == ord('t'):
      print("t")
      # pinto ventana donde muestro la cara y pido nombre
      # añado a la base de datos con ese nombre
      # ¿guardo la cara para futuro?
      
      if len(no_identificate_image) > 1 or len(no_identificate_image) < 1:
        print("Must be capture a UNKNOWN face of your image")
        text_info = "Must be capture a UNKNOWN face of your image"
        tm = 250
      else:
        # root = tk.Tk()
        # show input dialogs without the Tkinter window
        # root.withdraw()
        # nombre = tksd.askstring("identifciado", "Nombre de la persona capturada:")
        nombre = input("Your name (for example: peter_1): ")
        foto = no_identificate_image[0][0]
        cv2.imwrite(os.path.join(PATH_TO_DATABASE, nombre + '.png'), foto)
        # del root
    if key_pressed == ord('q'):
      # When everything done, release the capture
      cap.release()
      cv2.destroyAllWindows()
      break
