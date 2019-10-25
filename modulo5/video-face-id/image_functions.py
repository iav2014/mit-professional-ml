import cv2
import numpy as np

# API FINAL:
# extraeFeaturesImagenFromFile -> De un fichero de una única cara, extraer features
# extraeFeaturesImagen -> De una imagen abierta con cv2, extraer features
# extraeFeaturesRAW -> De una cara ya separada con dimensiones 96, 96, 3, extrae features
# who_is_it_single -> De una imagen abierta con cv2, obtener el match
# who_is_multiple -> De una imagen abierta con cv2, obtener los listados de matchs y no matchs
# who_is_it_RAW -> De una cara ya preparada (96, 96, 3), decir quién es
# identificaCaras ->De una imagen abierta con cv2, extraer las caras y matchs (identificados, fotos_caras, fotos_no_identificados)

PATH_TO_KERAS_MODEL = 'model\inception.h5'

TARGET_SIZE = (96, 96)
DEFAULT_THRESHOLD = 0.5


# preparaImagenCOGE LA IMAGEN y la traspone, por ejemplo, de 96,96,3 a 3,96,96
def prepareImage(img1):
  img = img1[..., ::-1]
  img = np.around(np.transpose(img, (2, 0, 1)) / 255.0, decimals=12)
  return img


# extraeUnaCaraEXTRAE UNA CARA DE UNA IMAGEN Y LA REDIMENSIONA A TARGET_SIZE
def getOneFace(imagen):
  face_cascade = cv2.CascadeClassifier('classifiers/haarcascade_frontalface_alt.xml')
  gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
  faces = face_cascade.detectMultiScale(gray, 1.3, 5)
  if len(faces) > 1:
    print("More than one faces")
    return None
  else:
    for (x, y, w, h) in faces:
      roi = imagen[y:y + h, x:x + w]
      roi = cv2.resize(roi, TARGET_SIZE)
      return roi
  return None


# extraeMultiplesCaras EXTRAE VARIAS CARAS DE UNA IMAGEN Y LAS REDIMENSIONA A TARGET_SIZE
def getMultiFace(imagen):
  face_cascade = cv2.CascadeClassifier('classifiers/haarcascade_frontalface_alt.xml')
  gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
  faces = face_cascade.detectMultiScale(gray, 1.3, 5)
  caras = []
  if len(faces) > 0:
    for (x, y, w, h) in faces:
      roi = imagen[y:y + h, x:x + w]
      roi = cv2.resize(roi, TARGET_SIZE)
      caras.append((roi, x, y, w, h))  # caras.append(roi)
    return caras
  return None


# preparaCaraDesdeImagen Extrae una cara, la redimensiona y la prepara
def getFaceFromImage(img1):
  resultado = None
  cara = getOneFace(img1)
  if cara is not None:
    resultado = prepareImage(cara)
  return resultado


# extraeFeaturesImagen De una única imagen extrae las features
def getImageFeature(img1, modelo):
  resultado = getFaceFromImage(img1)
  if resultado is not None:
    features = modelo.predict(np.array([resultado]))
    return features[0]
  return None


# extraeFeaturesImagenFromFile abro la imagen con opencv
def getFeatureFromFile(path, modelo):
  imagen = cv2.imread(path, 1)
  return getImageFeature(imagen, modelo)


# getFeaturesRAW De una única imagen extrae las features
def getFeaturesRAW(img1, modelo):
  d1, d2, d3 = img1.shape
  if d1 == 96 and d2 == 96 and d3 == 3:
    resultado = prepareImage(img1)
    features = modelo.predict(np.array([resultado]))
    return features[0]
  return None


# Si ya tenemos la cara sacada, directamente sacamos el vector de predicciones
# OJO, esta es sólo para usar cuando ya hemos extraído las caras de las fotos, y tiene el tamaño adecuado
def extraeFeaturesCara(img1, modelo):
  features = modelo.predict(np.array([resultado]))
  return features[0]


# De una imagen cualquiera, decir quién es
def who_is_it_single(imagen, database, model, threshold=None, debug=True):
  if threshold is None:
    threshold = 0.7
  
  ## Step 1: extracción de features
  encoding = getImageFeature(imagen, model)
  
  if encoding is None:
    print("ERROR, no hemos reconocido cara en la imagen")
    return None, None
  ## Step 2: Find the closest encoding ##
  
  # Initialize "min_dist" to a large value, say 100 (≈1 line)
  min_dist = 100
  identity = None
  
  # Loop over the database dictionary's names and encodings.
  for (name, db_enc) in database.items():
    
    # Compute L2 distance between the target "encoding" and the current "emb" from the database. (≈ 1 line)
    dist = np.linalg.norm(db_enc - encoding)
    
    # If this distance is less than the min_dist, then set min_dist to dist, and identity to name. (≈ 3 lines)
    if dist < min_dist:
      min_dist = dist
      identity = name
  
  if min_dist > threshold:
    if debug == True:
      print("Not in the database.")
    return None, None
  else:
    if debug == True:
      print("it's " + str(identity) + ", the distance is " + str(min_dist))
    return min_dist, identity


# De una cara ya preparada (96, 96, 3), decir quien es
def who_is_it_RAW(imagen, database, model, threshold=None, debug=True):
  if threshold is None:
    threshold = 0.7
  
  ## Step 1: extracción de features
  encoding = getFeaturesRAW(imagen, model)
  
  if encoding is None:
    print("ERROR, no hemos reconocido cara en la imagen")
    return None, None
  ## Step 2: Find the closest encoding ##
  
  # Initialize "min_dist" to a large value, say 100 (≈1 line)
  min_dist = 100
  identity = None
  
  # Loop over the database dictionary's names and encodings.
  for (name, db_enc) in database.items():
    
    # Compute L2 distance between the target "encoding" and the current "emb" from the database. (≈ 1 line)
    dist = np.linalg.norm(db_enc - encoding)
    
    # If this distance is less than the min_dist, then set min_dist to dist, and identity to name. (≈ 3 lines)
    if dist < min_dist:
      min_dist = dist
      identity = name
  
  if min_dist > threshold:
    if debug == True:
      print("Not in the database.")
    return None, None
  else:
    if debug == True:
      print("it's " + str(identity) + ", the distance is " + str(min_dist))
    return min_dist, identity


def faceID(img, modelo, database, threshold=DEFAULT_THRESHOLD, debug=True):
  caras = getMultiFace(img)
  totales = 0
  identificados = []
  fotos_caras = []
  fotos_no_identificados = []
  if caras is None:
    return [], [], []
  for c in caras:
    dist, identi = who_is_it_RAW(c[0], database, modelo, threshold=threshold, debug=debug)
    if identi is not None:
      identificados.append(identi + ' (' + str(dist) + ')')
      fotos_caras.append((np.uint8(c[0]), c[1], c[2], c[3], c[4]))
      totales += 1
    else:
      fotos_no_identificados.append((np.uint8(c[0]), c[1], c[2], c[3], c[4]))
  return identificados, fotos_caras, fotos_no_identificados


import matplotlib.pyplot as plt


# ESTO SOLO PARA JUPYTER
# %matplotlib inline

def pintaImagenRAW(img1):
  img_RGB = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
  plt.imshow(img_RGB)
  plt.show()
