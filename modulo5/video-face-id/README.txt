h5 - Inception video face id demo using inception h5
tensorflow + keras - opencv
====================================================

This example of realtime video recognition using a inception model.
We used a CNN using Stochastic Gradient Descent (SGD) with standard backprop (see modulo 5 MIT for more info)
details at https://arxiv.org/pdf/1503.03832.pdf

opencv
We use opencv to read and write images, to get images from webcam and detect 'face' using the xml classifiers



1- install python3 (download pip distribution)

pip3 install tensorflow --upgrade

2- install pip modules
pip install -r dependencies.txt

3- execute start.py

python start.py

	key control:(mouse focus must be in webcam window)
	
	l -> (LOAD), load image from dataset and create model in memory
	t -> take image from webcam, and put id name and save in database
	a -> + threshold
	z -> - threshold
	q -> quit

Steps: when the webcam image appears in window,
1) your face must me focused with "Unknown" title

2) press "t" key to take your image:

Your name (for example: nacho_01): nacho_01

take several images [5 to 10] per user, and named like this: [user_01 .. user_10]

it is important to take several photos of the same user, in order to improve the classification.

3) train and load the model in memory, press "l" key

4) the system will be recognize your image putting the filename up of your face into a green rectangle.

5) you can add images of different people to recognize, taking his image and reloading the model at the same session (press "l" key

to added to the model)

you can play using "a" key to up threshold and "z" key to down the threshold.

The model doesnt save on disk and must be calculate every time to start the program (pressing "l" key)


Authors
Javi Gomez (jun 2018) & Nacho Ariza (oct 2019)

Thanks to Skuldur for training the inception.h5 model

see: https://github.com/Skuldur/facenet-face-recognition

Based in this paper: https://arxiv.org/pdf/1503.03832.pdf

MIT License

