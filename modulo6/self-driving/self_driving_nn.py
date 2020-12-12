## simple neural network to control basic little Mars Rover (with arduino??)
# 4 servo wheels,
#                  center infrared
#  left infrared        /-|-\ right infrared
#  front left wheel [|]-------[|] front right wheel
#  rear left wheel  [|]-------[|] rear right wheel
#
#  inputs:
#  infrared: left 1000, too near left : 1100
#  infrared: center: 0110, too near center : 1111
#  infrared: right: 0001, too near right: 0011
#  output
#  wheels: forward: 1001 , reverse: 0110
#  turn left: 1010 turn right: 0101
#
# author: Nacho Ariza, oct 2019
## MIT License

import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
from keras.models import model_from_json
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator

EPOCHS=500
BS=128
# sensor data inputs (16)
sensor_data = np.array([[0, 0, 0, 0], # sin obstaculo / no obstacle ahead
                          [0, 0, 0, 1], # obstaculo a la derecha / right  obstacle
                          [0, 0, 1, 0], # sin obstaculo / no obstacle ahead
                          [0, 0, 1, 1], # obstaculo demasiado cerca de la derecha
                          [0, 1, 0, 0], # sin obstaculo / no obstacle ahead
                          [0, 1, 0, 1], # sin obstaculo / no obstacle ahead
                          [0, 1, 1, 0], # obstaculo enfrente / obstacle ahead
                          [0, 1, 1, 1], # sin obstaculo / no obstacle ahead
                          [1, 0, 0, 0],# obstaculo a la izquierda /right obstacle
                          [1, 0, 0, 1],# sin obstaculo / no obstacle ahead
                          [1, 0, 1, 0],# sin obstaculo / no obstacle ahead
                          [1, 0, 1, 1],# sin obstaculo / no obstacle ahead
                          [1, 1, 0, 0],# obstaculo demasiado cerca de la izquierda / obstacle near left
                          [1, 1, 0, 1],# sin obstaculo / no obstacle ahead
                          [1, 1, 1, 0],# sin obstaculo / no obstacle ahead
                          [1, 1, 1, 1] # obstaculo demasiado cerca del centro / obstacle near ahead
                          ], "float32")

# action data output (16)
action_data = np.array([[1, 0, 0, 1],  #avanzar / forward
                        [0, 1, 0, 1], # giro a la derecha /turn right
                        [1, 0, 0, 1], # avanzar / forward
                        [0, 1, 1, 0], # retroceder / back
                        [1, 0, 0, 1], # avanzar / forward
                        [1, 0, 0, 1], # avanzar /forward
                        [0, 1, 1, 0], # retroceder / back
                        [1, 0, 0, 1], # avanzar / forward
                        [1, 0, 1, 0], # giro a la izquierda /turn left
                        [1, 0, 0, 1], # avanzar / forward
                        [1, 0, 0, 1],# avanzar / forward
                        [1, 0, 0, 1],# avanzar / forward
                        [0, 1, 1, 0], # retroceder / back
                        [1, 0, 0, 1], # avanzar / forward
                        [1, 0, 0, 1], # avanzar / forward
                        [0, 1, 1, 0] # retroceder / back
                        ], "float32")

from sklearn import preprocessing

min_max_scaler = preprocessing.MinMaxScaler()

# neural network architecture
model = Sequential()
model.add(Dense(16, input_dim=4, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(4, activation='sigmoid'))

#compile model
model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['binary_accuracy'])

# fit model
history=model.fit(sensor_data, action_data, epochs=EPOCHS, batch_size=BS)

# evaluate model wit same inputs
scores = model.evaluate(sensor_data, action_data)

print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

# predict
print(model.predict(sensor_data).round())
# this only use if you wants to save trained model later
model_json = model.to_json()
with open("sequential.model.json", "w") as json_file:
    json_file.write(model_json)
# save weights to model.h5 file
model.save_weights("sequential.model.h5")
print("model saved!")

# if you wants to use the model later,

# load json & model
json_file = open('sequential.model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("sequential.model.h5")
print("model loaded!")

# compile model,
loaded_model.compile(loss='mean_squared_error', optimizer='adam', metrics=['binary_accuracy'])
# and now you can use the model...

# paint loss function and accuracy train graphics,,,
print(history.history.keys())
plt.plot(history.history['loss'])
plt.plot(history.history['binary_accuracy'])
plt.title('model accuracy')
plt.ylabel('binary accuracy')
plt.xlabel('epoch')
plt.legend(['loss train', 'test train'], loc='upper left')
plt.show()