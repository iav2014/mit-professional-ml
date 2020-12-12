CNN - VGGNET diagnostic classifier
==================================

Breast Ultrasound Dataset
Dataset provided and classificated by:
Dr.Moi Hoon Yap <M.Yap@mmu.ac.uk>
Dr.Robert Marti <robert.marti@udg.edu>
request date: 03-07-2019
Yap, M.H., Pons, G., Marti, J., Ganau, S., Sentis, M., Zwiggelaar, R., Davison, A.K. and Marti, R.(2017),
Automated Breast Ultrasound Lesions Detection using Convolutional Neural Networks.
IEEE journal of biomedical and health informatics. doi: 10.1109/JBHI.2017.2731873

Dr. Moi Hoon Yap
Reader in Computer Vision
Lead, Human-Centred Computing
Address:
Manchester Metropolitan University | John Dalton Building (E129) | Chester Street | Manchester | M1 5GD
Website: http://www2.docm.mmu.ac.uk/STAFF/M.Yap/
disclaimer: http://www.mmu.ac.uk/emaildisclaimer

CNN code architecture: https://www.pyimagesearch.com/

Diagnosis system based on neural networks, for detection of breast cancer.

Disclaimer
----------
This work is for didactic purposes and should not be used for real diagnosis


Requirements
--------------
-  Install `tensorflow + keras`_

https://www.tensorflow.org/install
https://keras.io/

-  opencv
https://opencv.org/



Train: generate model
---------------------
Estimated time: 7 min
macbookpro 2017, 15" 16Gb i7 ssd

::

    $ python3 train.py -d dataset -m model/medical.model -l labels/labels-medical -p metrics/plt-medical.png

Test & classify
---------------
for classification see: dataset.xlsx
test bening/malignant

::
    $ python3 classify.py -m model/medical.model -l labels/labels-medical -i test/malignant/000023-Malignant-DCIS.png
    $ python3 classify.py -m model/medical.model -l labels/labels-medical -i test/benign/000130-Benign-CYST.png


License
-------
MIT license
MIT Professional education 2019
Nacho Ariza sep.2019


