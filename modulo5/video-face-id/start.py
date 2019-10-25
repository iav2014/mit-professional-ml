from sys import platform as sys_pf

if sys_pf == 'darwin':
  import matplotlib
  
  matplotlib.use("TkAgg")
from take_image import *
from inception import *

# constants
PATH_TO_KERAS_MODEL = 'model/inception.h5'
TARGET_SIZE = (96, 96)
DEFAULT_THRESHOLD = 0.5
WEBCAM = 0
PATH_TO_DATABASE = 'images'

# help menu
print("l -> Load model")
print("t -> take new face")
print("a -> increase precision")
print("z -> decrease precision")
print("q -> quit")

# webcam image loop
modelo = loadModel(PATH_TO_KERAS_MODEL)
capture(modelo, dict())
