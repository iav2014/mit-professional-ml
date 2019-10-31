RNN -  self driving example model (little mars rover)
=====================================================

Watching the video of the module6 that talk about the mars rover, 
it occurred to me to apply the concept of neural networks to autonomous driving.

This is an example of how a neural network can help us make a predictive model 
that can govern a car using for example arduino. 

The described model has a series of input data that are the data that 
come from the ultrasonic sensors of the vehicle (3), which rotate 
from east to west, continuously. 

The output of the neural network simulates the orders to be given 
to the wheel motors, in order to avoid obstacles.

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



Disclaimer
----------
This work is for didactic purposes and should not be used for real driving!


Requirements
--------------
-  Install `tensorflow + keras`_

https://www.tensorflow.org/install
https://keras.io/

Train: generate model
---------------------
Estimated time: 10 sec
macbookpro 2017, 15" 16Gb i7 ssd
Epoch 500/500

16/16 [==============================] - 0s 183us/step - loss: 0.0161 - binary_accuracy: 1.0000

16/16 [==============================] - 0s 2ms/step

binary_accuracy: 100.00%

::

    $ python3 self_driving_nn.py


if you wants to create && see the car (not using this RNN):

https://www.youtube.com/watch?v=4CFO0MiSlM8

License
-------

MIT Professional education 2019
Nacho Ariza oct.2019


