'''

6.2.18

IT ACTUALLY WORKS.

Attempting to load trained model and make a prediction.

'''

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

from keras.preprocessing.image import img_to_array, load_img

# model.save option
from keras.models import load_model
from keras.preprocessing import image
import numpy as np


import cv2
import glob
from keras.preprocessing.image import img_to_array

# Data preprocessing
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

# DATASET PREPERATION

# initialize the data and labels
print("[INFO] loading images...\n")
data = []
labels = []

TestImagePath = sorted(list((glob.glob("/home/olly/projects/machine_learning/catsvsdogs2/test/*.jpg"))))

for TestImage in TestImagePath:

	image = cv2.imread(TestImage)

	image = cv2.resize(image, (150, 150))

	# Let's look at some lovely cats
	cv2.imshow("Test", image)
	cv2.waitKey(0)

	# Store image in data list
	image = img_to_array(image)
	data.append(image)


# additional data prep you silly sausage
data = np.array(data, dtype="float") / 255.0







# load the model we saved

from keras.models import model_from_json
# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")


loaded_model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# make predictions
sample = 0 #choosing sample from testing dataset


x_sample = data[sample].reshape(1, 3, 150, 150) #X_test.shape[0]
y_prob = loaded_model.predict(x_sample)#[0]
y_pred = y_prob.argmax()
#y_actual = Y_test[sample].argmax()

print("probabilities: ", y_prob)
print("predicted =" , y_pred)
#print("predicted = %d, actual = %d" % (y_pred, y_actual))
if y_pred == 0:
	print "Cat!"
else:
	print "Dog!"

print "\n Testing predictions\n"

sample = data[0]
print "fresh cat? prediction"
test_array = sample.reshape(1, 3,150 ,150)
prediction = loaded_model.predict(test_array)
print prediction
y_pred = prediction.argmax()
print("predicted =" , y_pred)
if y_pred == 1:
	print "Cat!"
else:
	print "Dog!"


sample = data[1]
print "fresh dog? prediction"
test_array = sample.reshape(1, 3,150 ,150)
prediction = loaded_model.predict(test_array)
print prediction
y_pred = prediction.argmax()
print("predicted =" , y_pred)
if y_pred == 1:
	print "Cat!"
else:
	print "Dog!"
'''

		IT		FINALLY		WORKS

'''













