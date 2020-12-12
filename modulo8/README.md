RNN -  time_series example
=====================================================
MIT modulo8 - time series

example of use for time series. Make a one-week forecast 
on the net asset value of an investment fund, based on the 
fund's record of the last 2 years. 

ToRead
======
https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/
https://www.tensorflow.org/tutorials/structured_data/time_series
https://towardsdatascience.com/time-series-forecasting-with-recurrent-neural-networks-74674e289816



Inputs
======
prices.csv , values x day x > 2 years

Neural network used
===================
7 inputs, 1 hidden (7 neurons) , 1 output

  model = Sequential()
  model.add(Dense(7, input_shape=(1, 7), activation='tanh'))
  model.add(Flatten())
  model.add(Dense(1, activation='tanh'))
  model.compile(loss='mean_absolute_error', optimizer='Adam', metrics=["mse"])
  model.summary()

Disclaimer
----------
This work is for didactic purposes and should not be used for real purposes!

Requirements
--------------
-  Install `tensorflow + keras`_

https://www.tensorflow.org/install
https://keras.io/

Train: generate model
---------------------
Estimated time: 10 sec
macbookpro 2017, 15" 16Gb i7 ssd
Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_1 (Dense)              (None, 1, 7)              56        
_________________________________________________________________
flatten_1 (Flatten)          (None, 7)                 0         
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 8         
=================================================================
Total params: 64
Trainable params: 64
Non-trainable params: 0
_________________________________________________________________
Train on 567 samples, validate on 399 samples
Epoch 1/70

  7/567 [..............................] - ETA: 7s - loss: 0.9903 - mse: 1.1404
462/567 [=======================>......] - ETA: 0s - loss: 0.5338 - mse: 0.4006
567/567 [==============================] - 0s 361us/step - loss: 0.4551 - mse: 0.3309 - val_loss: 0.4391 - val_mse: 0.2145
Epoch 2/70


::

    $ python3 time_series.py

Prediction
==========
[[78.12506785]
 [78.05761503]
 [78.03747893]
 [77.99229909]
 [78.12102971]
 [77.95720875]
 [78.07248281]]



License
-------

MIT Professional education 2019
Nacho Ariza nov.2019


